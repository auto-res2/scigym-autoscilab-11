"""1 件を LLM-AutoSciLab の部品で解く（論文 Algorithm 1）。

上流をそのまま呼ぶ: EnsembleAgent.run_round（GenHyp: small の K 並列サンプル → 骨格クラスタ → large の整理）、
FalsificationSelector.select（Acquire）、LLMMemory、ExperimentStore。
Python 式が前提の 2 メソッド（sample_hypothesis_distribution / _sample_adaptive）だけ Antimony 用に上書きする。
SBML 固有の段階（canonical 化、定数フィット、確信度、Refine のスコア、提出）は src/model.py。このループは部品を順に呼ぶだけ。
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from scigym.actions.experiment import Experiment
from scigym.data.question import Question

from autoscilab.al.falsification import FalsificationSelector
from autoscilab.llm.client import LLMClient
from autoscilab.llm.ensemble_agent import EnsembleAgent
from autoscilab.llm.ensemble_client import EnsembleClient
from autoscilab.llm.memory import LLMMemory
from autoscilab.llm.prompts import SYSTEM_PROMPT, build_hypothesis_generation_prompt

from src import keys as K
from src.model import (Candidate, SciGymOracle, System, TrajectoryDistribution, TrajectoryStore, bootstrap_confidence, fit,
                       log_r2, make_candidate, make_rng, refine_gate, set_sim_timeouts)

MAX_HYPOTHESES = 8  # 1 反復で持つ仮説集合 H_t の上限（本命 + 対抗案）
GOAL = ("Recover the reactions that were removed from the SBML model: propose the complete set of reactions (with rate laws and "
        "rate constants) that reproduces the measured concentration time series of every species under every perturbation.")


class AntimonyBundle(BaseModel):
    """small モデルの出力。上流の HypothesisBundle は Python 式を要求するので、Antimony の反応ブロックに差し替える"""

    reasoning: str = Field(description="Reasoning about which reactions and rate laws the data supports.")
    hypothesis: str = Field(description="ONE complete hypothesis as an Antimony reaction block: every missing reaction, each followed by "
                            "its rate law, then one assignment per rate constant. No prose, no species assignments.")
    alternate_hypotheses: list[str] = Field(default_factory=list, max_length=6, description="2-6 distinct alternative Antimony blocks.")


def instructions(system: System) -> str:
    """オラクルの param_description として上流の prompt に入る、課題と Antimony の書き方"""
    species = "\n".join(f"  - {s}: initial concentration {system.initial.get(s, 0.0):.4g}" for s in system.species)
    comps = ", ".join(system.compartments)
    return f"""The system is an SBML model whose reactions were all removed; only species and compartments remain.
Species ids (use exactly these):
{species}
Compartments: {comps}
A hypothesis is an Antimony reaction block. Rules:
  - one reaction per statement: `name: reactants -> products; rate_law;` (use `=>` for irreversible, `->` for reversible);
    an empty side is allowed (`-> id_x` synthesis, `id_x =>` degradation); stoichiometry as `2 id_x`.
  - the rate law is any mathematical expression of species ids and named constants, multiplied by the compartment id
    (e.g. `{system.compartments[0]} * k1 * id_a * id_b / (K + id_a)`); mass action, saturating, Hill, products or differences of terms are all fine.
  - give every constant a value on its own statement: `k1 = 0.5;` (values are refitted to the data afterwards).
  - never assign initial concentrations of species or compartment sizes; only reactions and constants.
  - the block must contain ALL reactions the system needs, since the model has none.
The experiments vary initial concentrations of species (the "parameters" below are species ids) and return the full time series."""


class SBMLEnsembleAgent(EnsembleAgent):
    """EnsembleAgent のまま。仮説を Python 式として選別・クラスタする 2 メソッドだけ Antimony 用に上書きする（上流 s2.p29）"""

    def __init__(self, *args, system: System, **kwargs):  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.system = system
        self.candidates: dict[str, Candidate] = {}  # canonical -> Candidate（同じ骨格は 1 つの候補に束ねる）
        self.rejected = 0

    def candidate(self, block: str) -> Candidate | None:
        """ブロック → 候補。壊れている・回らないものは捨てる（上流の FilterValid に当たる）"""
        try:
            cand = make_candidate(block, self.system)
        except Exception:
            self.rejected += 1
            return None
        return self.candidates.setdefault(cand.canonical, cand)

    def _sample_adaptive(self, messages: list[dict]) -> list[str]:
        """上流と同じ Algorithm 2: K ずつサンプルし、骨格クラスタのエントロピーが落ち着くか K_max で止める。骨格は canonical 化した Antimony"""
        k_batch = self._ensemble._k
        canonicals: list[str] = []
        prev_entropy = None
        total = 0
        while total < self._k_max:
            batch = min(k_batch, self._k_max - total)
            try:
                bundles = self._ensemble.complete_json_k(messages=messages, system=SYSTEM_PROMPT, schema=AntimonyBundle, k_override=batch)
            except RuntimeError:  # 1 つも JSON にならなかった
                bundles = []
            total += batch
            for b in bundles:
                c = self.candidate(b.hypothesis)
                if c is not None:
                    canonicals.append(c.canonical)
            log(f"small batch: {len(bundles)} parsed, {len(canonicals)} valid so far, {self.rejected} rejected in total")
            if len(canonicals) < k_batch:
                continue
            h = TrajectoryDistribution.from_expressions(canonicals).entropy()
            if prev_entropy is not None and abs(h - prev_entropy) < self._stability_threshold:
                break
            prev_entropy = h
        return canonicals

    def sample_hypothesis_distribution(self, goal, store, current_hypothesis, budget_remaining, total_budget):  # noqa: ANN001, ANN201
        prompt = build_hypothesis_generation_prompt(
            goal=goal, domain_id=self._oracle.domain, param_names=self._oracle.parameter_names,
            param_description=self._oracle.param_description, function_signature=self._oracle.function_signature,
            data_table=store.to_text_table(max_rows=30), current_hypothesis=current_hypothesis,
            budget_remaining=budget_remaining, total_budget=total_budget, current_phase="discovery",
            best_equation_fit=None, memory_str=self._memory.to_prompt_str(), slope_hints="",
            objective_type="equation", objective_direction="maximize", objective_profile={})
        self._call_count += 1
        messages = [{"role": "user", "content": prompt}]
        if self._adaptive:
            canonicals = self._sample_adaptive(messages)
        else:
            bundles = self._ensemble.complete_json_k(messages=messages, system=SYSTEM_PROMPT, schema=AntimonyBundle)
            canonicals = [c.canonical for c in (self.candidate(b.hypothesis) for b in bundles) if c is not None]
        dist = TrajectoryDistribution.from_expressions(canonicals)
        self._last_distribution = dist
        log(f"sampled {len(canonicals)} hypotheses, {dist.n_unique_structures()} unique structures, entropy {dist.entropy():.2f} bits")
        return dist


class TokenCounter:
    """公式クライアントが内部で持つ OpenAI クライアントの create を包んで usage を足し上げる（上流コードは変えない）"""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def _add(self, response) -> None:  # noqa: ANN001
        usage = getattr(response, "usage", None)
        self.calls += 1
        if usage is not None:
            self.input_tokens += usage.prompt_tokens or 0
            self.output_tokens += usage.completion_tokens or 0

    def wrap(self, completions) -> None:  # noqa: ANN001
        original = completions.create

        async def counted(coro):  # noqa: ANN001
            r = await coro
            self._add(r)
            return r

        def create(*a, **kw):  # noqa: ANN002, ANN003
            r = original(*a, **kw)
            # AsyncCompletions.create は required_args で包まれていて coroutine 関数に見えないので、戻り値で判定する
            if inspect.iscoroutine(r):
                return counted(r)
            self._add(r)
            return r
        completions.create = create


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Loop:
    def __init__(self, cfg: dict, out: Path):
        self.cfg, self.out = cfg, out
        self.rng = make_rng(cfg[K.SEED])
        set_sim_timeouts(cfg[K.SIM_TIMEOUT], cfg[K.SIM_TIMEOUT_PER_JOB])
        inst = cfg["instance_dir"]
        q = Question(sbml_directory_path=inst, task_difficulty="fully_observable")
        complete, incomplete = q.get_original_sbml(), q.get_partial_sbml()
        self.system = System.from_incomplete(incomplete, Path(inst, "truth.sedml").read_text())
        self.allowed = self.system.allowed_bounds(tuple(cfg[K.BOUNDS_FACTOR]))
        experiment = Experiment(true_sbml=copy.deepcopy(complete), inco_sbml=copy.deepcopy(incomplete))
        self.oracle = SciGymOracle(self.system, experiment, dict(self.allowed), instructions(self.system))
        self.store = TrajectoryStore("scigym", list(self.allowed), self.system.species, self.system.n_points)
        self.memory = LLMMemory()
        key = os.environ["VERCEL_AI_GATEWAY_API_KEY"]
        small = EnsembleClient(base_url=cfg[K.ENSEMBLE_BASE_URL], model=cfg[K.ENSEMBLE_MODEL], k=int(cfg[K.ENSEMBLE_K]),
                               temperature=float(cfg[K.ENSEMBLE_TEMPERATURE]), max_tokens=int(cfg[K.ENSEMBLE_MAX_TOKENS]), api_key=key)
        large = LLMClient(model=cfg[K.LARGE_MODEL], base_url=cfg[K.LARGE_BASE_URL], api_key=key,
                          max_completion_tokens=int(cfg[K.LARGE_MAX_COMPLETION_TOKENS]))
        self.tokens = TokenCounter()
        self.tokens.wrap(small._client.chat.completions)
        self.tokens.wrap(large._oa_client.chat.completions)
        self.agent = SBMLEnsembleAgent(small, large, self.oracle, self.memory, adaptive=bool(cfg[K.AGENT_ADAPTIVE]), k_max=int(cfg[K.AGENT_K_MAX]),
                                       stability_threshold=float(cfg[K.AGENT_STABILITY_THRESHOLD]), system=self.system)
        self.best: Candidate | None = None
        self.trace: list[dict] = []
        self.proposal_valid = 0

    # ---------------------------------------------------------------- pieces around the upstream calls
    def hypotheses(self, dist, proposal) -> list[Candidate]:  # noqa: ANN001
        """H_t: 骨格クラスタの代表（大きい順）+ large が書いた本命 + これまでの最良"""
        hyps: list[Candidate] = []
        proposed = self.agent.candidate(proposal.hypothesis) if proposal and proposal.hypothesis else None
        if proposed is not None:
            self.proposal_valid += 1
            hyps.append(proposed)
        for entry in dist.unique_hypotheses:
            c = self.agent.candidates.get(entry.expression)
            if c is not None and c not in hyps:
                hyps.append(c)
        if self.best is not None and self.best not in hyps:
            hyps.insert(0, self.best)
        return hyps[:MAX_HYPOTHESES]

    def region(self, proposal) -> dict[str, tuple[float, float]]:  # noqa: ANN001
        """large が提案した探索領域（上流の SearchRegion）を許容範囲に切り詰める。無ければ許容範囲全体"""
        region: dict[str, tuple[float, float]] = {}
        for r in (proposal.search_regions if proposal else []):
            for s, b in r.bounds.items():
                if s in self.allowed and isinstance(b, (list, tuple)) and len(b) == 2:
                    lo, hi = self.allowed[s]
                    a, c = max(lo, min(hi, float(b[0]))), max(lo, min(hi, float(b[1])))
                    if c > a:
                        region[s] = (min(region[s][0], a), max(region[s][1], c)) if s in region else (a, c)
        return region or dict(self.allowed)

    def experiment(self, initial: dict[str, float]) -> None:
        try:
            self.store.add(self.oracle.run(initial))
        except RuntimeError as exc:  # 失敗した実験も予算を消費する（SciGym と同じ）
            log(f"experiment failed: {str(exc)[:100]}")

    # ---------------------------------------------------------------- Algorithm 1
    def run(self) -> None:
        cfg = self.cfg
        budget, per_iter = int(cfg[K.TOTAL_BUDGET]), int(cfg[K.SELECT_N])
        self.experiment({})  # 最初の 1 回は既定条件の観測
        current, confidence, refits, rounds = "unknown", 0.0, [], 0
        while self.oracle.n_calls < budget and rounds < 2 * budget:
            rounds += 1
            valid_before = self.proposal_valid
            proposal, current, dist = self.agent.run_round(goal=GOAL, store=self.store, current_hypothesis=current,
                                                           budget_remaining=budget - self.oracle.n_calls, total_budget=budget,
                                                           best_equation_fit=self.best.block if self.best else None)
            hyps = self.hypotheses(dist, proposal)
            if not hyps:
                log("no valid hypothesis this round")
                continue
            for i, c in enumerate(hyps):
                fit(c, self.store, rng=self.rng, restarts=1 if math.isinf(c.fit_error) else 0)
                log(f"fit {i + 1}/{len(hyps)}: {c.fit_error:.3f}")
            hyps.sort(key=lambda c: c.fit_error)
            leader = hyps[0]
            confidence, refits = bootstrap_confidence(leader, self.store, int(cfg[K.N_BOOTSTRAP]), self.rng, float(cfg[K.MIN_R2]))
            refine = refine_gate(confidence, cfg[K.CONFIDENCE_THRESHOLD], len(refits))
            self.oracle.bounds = self.region(proposal)
            # Disambiguate: 仮説間の軌道の食い違い / Refine: 本命の再フィット間の不確実性。選択と多様性は同じ上流の select
            members = [(leader, p) for p in refits] if refine else [(c, c.params) for c in hyps]
            dist2 = TrajectoryDistribution.over(members, self.store, acquisition=cfg[K.ACQUISITION])
            selector = FalsificationSelector(self.oracle, dist2, n_candidates=int(cfg[K.N_CANDIDATES]), diversity_weight=float(cfg[K.DIVERSITY_WEIGHT]))
            points = selector.select(n=per_iter, rng=self.rng)  # 常に per_iter 点を選び、予算の残りだけ使う
            chosen = points[: budget - self.oracle.n_calls]
            for p in chosen:
                self.experiment(p)
            for c in hyps:
                fit(c, self.store, restarts=0, rng=self.rng)
            hyps.sort(key=lambda c: c.fit_error)
            self.best = min([self.best, hyps[0]] if self.best else [hyps[0]], key=lambda c: c.fit_error)
            current = self.best.block
            self.memory.add(iteration=rounds, n_data=len(self.store), equation=self.best.canonical,
                            r2=log_r2(self.best, self.store), rmsle=self.best.fit_error, bootstrap_conf=confidence)
            self.trace.append({
                "round": rounds, "mode": "refine" if refine else "disambiguate", "confidence": confidence,
                "hypotheses": [{"canonical": c.canonical, "fit_error": c.fit_error} for c in hyps],
                "primary_valid": self.proposal_valid > valid_before,
                "region": {s: list(b) for s, b in self.oracle.bounds.items()}, "experiments": chosen,
                "n_experiments": self.oracle.n_calls, "tokens": [self.tokens.input_tokens, self.tokens.output_tokens],
            })
            log(f"round {rounds}: {len(hyps)} hyps, best {self.best.fit_error:.3f} ({self.best.canonical[:90]}), "
                f"confidence {confidence:.2f} {'refine' if refine else 'disambiguate'}, experiments {self.oracle.n_calls}/{budget}")
        if self.best is not None:
            fit(self.best, self.store, rng=self.rng)
        self.finish(confidence, rounds)

    def finish(self, confidence: float, rounds: int) -> None:
        if self.best is not None:
            (self.out / "final_model.xml").write_text(self.best.sbml_with_params())
        (self.out / "trace.json").write_text(json.dumps(self.trace, indent=1, default=float))
        (self.out / "tokens.json").write_text(json.dumps({
            "input_tokens": self.tokens.input_tokens, "output_tokens": self.tokens.output_tokens, "llm_calls": self.tokens.calls,
            "n_experiments": self.oracle.n_calls, "n_rounds": rounds, "sim_worker_restarts": self.system.worker.restarts,
            "n_candidates_total": len(self.agent.candidates), "n_rejected_blocks": self.agent.rejected, "n_primary_valid": self.proposal_valid,
            "best_fit_error": self.best.fit_error if self.best else None, "final_confidence": confidence, "submitted": self.best is not None,
        }))
        self.system.worker.close()
        log(f"done: submitted={self.best is not None} experiments={self.oracle.n_calls} tokens={self.tokens.input_tokens}/{self.tokens.output_tokens}")


def main() -> None:
    cfg = json.loads(sys.argv[1])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    Loop(cfg, out).run()


if __name__ == "__main__":
    main()
