"""LLM-AutoSciLab の部品を SciGym に繋ぐアダプタ。仮説 = Antimony の反応ブロック（LLM が反応・速度則・定数を自由に書く）、観測 = 全種 × 全時刻の軌道。

上流（autoscilab）のクラスはそのまま使い、Python 式（スカラー）が前提になっている部分だけをサブクラスで差し替える:
- HypothesisDistribution: 骨格・クラスタ・エントロピーは上流のまま（canonical 化した Antimony を渡す）。compute_disagreement だけ軌道に
- ExperimentStore / OracleResult: 軌道を型付きで持ち、prompt 用の表だけ軌道の要約に
- BaseOracle.run: SciGym の Experiment.call_simulator へ
- _JsonWorker: 上流の request は readline でブロックするので、時間切れと作り直しを足す
`python src/model.py worker` で起動すると候補 SBML のシミュレーション子プロセスになる。
"""

from __future__ import annotations

import json
import math
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from scigym.data.sbml import SBML  # scigym（tellurium 経由の roadrunner）を libsbml より先に読む: 逆順だと二重の libsbml で落ちる
from scigym.data.simulator import Simulator, run_simulation

import libsbml  # noqa: E402
import numpy as np  # noqa: E402
import roadrunner  # noqa: E402
import tellurium as te  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

roadrunner.Logger.setLevel(roadrunner.Logger.LOG_FATAL)  # 候補の積分失敗は想定内。CVODE のエラー行でログを埋めない

WORKER = __name__ == "__main__"
if not WORKER:  # worker モードでは上流を読まない
    from autoscilab.data.store import ExperimentStore
    from autoscilab.llm.hypothesis_dist import HypothesisDistribution
    from autoscilab.oracle.base import BaseOracle, OracleResult
    from autoscilab.oracle.external import _JsonWorker
else:
    ExperimentStore = HypothesisDistribution = BaseOracle = OracleResult = _JsonWorker = object

PARAM_LOG_BOUNDS = (-12.0, 12.0)  # 自由定数の自然対数の範囲
MAX_FIT_POINTS = 50  # フィットと食い違い計算に使う時点数の上限
FAIL_RESIDUAL = 10.0  # シミュレーションが落ちた点の残差（log スケール）
SIM_TIMEOUT_BASE = 15.0  # 1 バッチの時間切れ（秒）= BASE + PER_JOB × 件数
SIM_TIMEOUT_PER_JOB = 0.02


def set_sim_timeouts(sim_timeout: float, sim_timeout_per_job: float) -> None:
    """run の設定を生の値で受けて時間切れの式の係数にする（この引数が観測され、yaml と比べられる）"""
    global SIM_TIMEOUT_BASE, SIM_TIMEOUT_PER_JOB
    SIM_TIMEOUT_BASE, SIM_TIMEOUT_PER_JOB = float(sim_timeout), float(sim_timeout_per_job)


def make_rng(seed: int) -> np.random.Generator:
    """default_rng は C 実装で引数が観測に残らないので、seed はここで受ける"""
    return np.random.default_rng(int(seed))


def refine_gate(confidence: float, confidence_threshold: float, n_refits: int) -> bool:
    """確信度の門: 本命の確信度が閾値以上で再フィットが 2 つ以上なら Refine モード"""
    return confidence >= float(confidence_threshold) and n_refits >= 2
ANTIMONY_FUNCTIONS = {"pow", "exp", "log", "ln", "log10", "sqrt", "abs", "sin", "cos", "tan", "piecewise", "min", "max", "floor", "ceil", "time"}


# ------------------------------------------------------------------ Antimony
def base_antimony(incomplete: SBML) -> str:
    """不完全モデルを Antimony に。仮説の反応ブロックはこの末尾（end の前）に入る"""
    return te.sbmlToAntimony(incomplete.to_string())


def assemble(base: str, block: str) -> str:
    head, _, _ = base.rstrip().rpartition("end")
    return head + "\n  // hypothesis\n" + block.strip() + "\nend\n"


def to_sbml(base: str, block: str) -> str:
    """Antimony → SBML。文法が壊れていれば antimony が例外を投げる"""
    return te.antimonyToSBML(assemble(base, block))


_REACTION = re.compile(r"^(?:(\w+)\s*:)?\s*(.*?)\s*(=>|->)\s*(.*?)$")
_ASSIGN = re.compile(r"^(\w+)\s*=\s*(.+)$")
_IDENT = re.compile(r"(?<![\w.])([A-Za-z_]\w*)(?![\w(])")


def canonicalize(block: str, species: list[str], compartments: list[str]) -> tuple[str, str]:
    """LLM の反応ブロック → (骨格の文字列, 既定値を補った正規化ブロック)。

    骨格: 反応を (反応物, 生成物, 速度則) の順に並べ、定数を出現順に c1, c2 … に付け直し、数値は残す
    （上流の _extract_skeleton が数値を C に置換する）。種・区画への代入は仮説ではないので拒否する。
    """
    known = set(species) | set(compartments)
    statements = [s.strip() for s in re.sub(r"//[^\n]*", "", block).replace("\n", " ").split(";")]
    statements = [s for s in statements if s]
    reactions: list[tuple[list[str], list[str], str, str]] = []  # (reactants, products, arrow, rate)
    assigned: dict[str, str] = {}
    i = 0
    while i < len(statements):
        m = _REACTION.match(statements[i])
        if m and ("->" in statements[i] or "=>" in statements[i]):
            if i + 1 >= len(statements):
                raise ValueError("reaction without a rate law")
            rate = statements[i + 1]
            if _ASSIGN.match(rate) or "->" in rate or "=>" in rate:
                raise ValueError("reaction without a rate law")
            reactions.append((_side(m.group(2)), _side(m.group(4)), m.group(3), re.sub(r"\s+", "", rate)))
            i += 2
            continue
        a = _ASSIGN.match(statements[i])
        if a:
            if a.group(1) in known:
                raise ValueError(f"assignment to species/compartment {a.group(1)}")
            assigned[a.group(1)] = a.group(2).strip()
            i += 1
            continue
        raise ValueError(f"unrecognised statement: {statements[i][:40]}")
    if not reactions:
        raise ValueError("no reaction")
    for r, p, _, _ in reactions:
        for s in r + p:
            if s not in species:
                raise ValueError(f"unknown species {s}")
    order = sorted(range(len(reactions)), key=lambda k: ("+".join(reactions[k][0]), "+".join(reactions[k][1]), reactions[k][3]))
    renamed: dict[str, str] = {}

    def rename(m: re.Match) -> str:
        name = m.group(1)
        if name in known or name in ANTIMONY_FUNCTIONS:
            return name
        renamed.setdefault(name, f"c{len(renamed) + 1}")
        return renamed[name]

    skeleton = "; ".join(f"{'+'.join(reactions[k][0])}{reactions[k][2]}{'+'.join(reactions[k][1])}|{_IDENT.sub(rename, reactions[k][3])}" for k in order)
    lines = [f"{' + '.join(reactions[k][0])} {reactions[k][2]} {' + '.join(reactions[k][1])}; {reactions[k][3]};".strip() for k in order]
    lines += [f"{name} = {assigned.get(name, '1')};" for name in renamed]  # 速度則の定数は必ず値を持たせる（未宣言は 1）
    return skeleton, "\n".join(lines)


def _side(text: str) -> list[str]:
    out = []
    for term in text.split("+"):
        term = term.strip()
        if not term:
            continue
        m = re.match(r"^(\d+(?:\.\d+)?)\s*\*?\s*([A-Za-z_]\w*)$", term)
        if m:
            out += [m.group(2)] * int(float(m.group(1)))
        elif re.match(r"^[A-Za-z_]\w*$", term):
            out.append(term)
        else:
            raise ValueError(f"bad species term {term}")
    return sorted(out)


# ------------------------------------------------------------------ simulation worker (child side)
def _simulate_one(sim: Simulator, params: dict, initial: dict, species: list[str], index) -> list | None:  # noqa: ANN001
    try:
        sim.prepare_simulation()
        for pid, v in params.items():
            sim._rr.setValue(pid, float(v))
        for sid, v in initial.items():
            sim._rr.setInitConcentration(sid, float(v), forceRegenerate=False)
        res = run_simulation(sim._rr, observed_species=species, observed_parameters=[],
                             rm_concentration_brackets=True, sed_simulation=sim.simulation)
        y = np.array([res.result[s] for s in species], dtype=float).T
        if not np.all(np.isfinite(y)):
            return None
        return (y if index is None else y[index]).tolist()
    except Exception:
        return None


def worker_main() -> None:
    """子プロセス本体: {"cmd":"load"|"run"} を JSONL で受け、{"ok":true,...} を返す（上流 _JsonWorker の約束）"""
    sims: dict[str, Simulator] = {}
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        req = json.loads(raw)
        try:
            if req["cmd"] == "load":
                sims[req["cid"]] = Simulator(SBML(req["sbml"], req["sedml"]))
                out = {"ok": True}
            elif req["cmd"] == "run":
                sim = sims[req["cid"]]
                out = {"ok": True, "results": [_simulate_one(sim, p, i, req["species"], req["index"]) for p, i in req["jobs"]]}
            else:
                out = {"ok": False, "error": f"unknown cmd {req['cmd']}"}
        except Exception as exc:
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if WORKER and sys.argv[1:] == ["worker"]:
    worker_main()
    sys.exit(0)


# ------------------------------------------------------------------ simulation worker (parent side)
class SimWorker(_JsonWorker):
    """上流の _JsonWorker に時間切れを足す。時間切れなら子を kill して作り直し、そのバッチは失敗扱い"""

    def __init__(self, sedml: str, species: list[str]):
        self.sedml, self.species = sedml, species
        self.sbml: dict[str, str] = {}
        self.loaded: set[str] = set()
        self.restarts = 0
        self._spawn()

    def _spawn(self) -> None:
        _JsonWorker.__init__(self, Path(sys.executable), Path(__file__), ["worker"])
        self.loaded = set()

    def _restart(self) -> None:
        self.restarts += 1
        _JsonWorker.close(self)  # 止まった子は待たずに terminate / kill
        self._spawn()

    def close(self) -> None:
        """stdin を閉じて子を自然に終わらせる（観測フックが記録を書き終えるまで最大 60 秒）。駄目なら上流の close"""
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.close()
                proc.wait(timeout=60)
            except Exception:
                pass
        _JsonWorker.close(self)

    def request(self, payload: dict, timeout: float = 60.0) -> dict:  # noqa: D102
        box: list = []

        def call() -> None:
            try:
                box.append(_JsonWorker.request(self, payload))
            except Exception as exc:  # 子のエラーは RuntimeError で返る（上流）
                box.append(exc)
        t = threading.Thread(target=call, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            self._restart()
            raise TimeoutError
        if isinstance(box[0], Exception):
            raise box[0]
        return box[0]

    def load(self, cid: str, sbml: str) -> None:
        """候補を読み込む。壊れた SBML は RuntimeError"""
        self.sbml[cid] = sbml
        self.request({"cmd": "load", "cid": cid, "sbml": sbml, "sedml": self.sedml}, 60.0)
        self.loaded.add(cid)

    def run(self, cid: str, jobs: list[tuple[dict, dict]], index) -> list[np.ndarray | None]:  # noqa: ANN001
        if not jobs:
            return []
        try:
            if cid not in self.loaded:
                self.load(cid, self.sbml[cid])
            out = self.request({"cmd": "run", "cid": cid, "jobs": jobs, "species": self.species,
                                "index": None if index is None else [int(i) for i in index]},
                               SIM_TIMEOUT_BASE + SIM_TIMEOUT_PER_JOB * len(jobs))
            return [None if y is None else np.array(y, dtype=float) for y in out["results"]]
        except (TimeoutError, RuntimeError, OSError):
            return [None] * len(jobs)


# ------------------------------------------------------------------ system / candidates
@dataclass
class System:
    """1 件分の問題: 不完全モデル、種、SED-ML、Antimony の土台、シミュレーション子プロセス"""

    incomplete: SBML
    sedml: str
    species: list[str]
    perturbable: list[str]
    initial: dict[str, float]
    compartments: list[str]
    n_points: int
    base: str
    worker: SimWorker
    existing_params: set[str]

    @classmethod
    def from_incomplete(cls, incomplete: SBML, sedml: str) -> "System":
        m = incomplete.model
        species = incomplete.get_species_ids()
        perturbable = [s for s in incomplete.get_species_ids(floating_only=True) if not m.getSpecies(s).getConstant()]
        probe = Simulator(incomplete)
        probe.prepare_simulation()
        n_points = probe.simulation.getNumberOfSteps() + 1
        comps = [m.getCompartment(i).getId() for i in range(m.getNumCompartments())]
        return cls(incomplete, sedml, species, perturbable, incomplete.get_initial_concentrations(), comps, n_points,
                   base_antimony(incomplete), SimWorker(sedml, species), set(incomplete.get_parameter_ids()))

    def fit_index(self) -> np.ndarray:
        return np.unique(np.linspace(0, self.n_points - 1, min(self.n_points, MAX_FIT_POINTS)).round().astype(int))

    def allowed_bounds(self, factor: tuple[float, float]) -> dict[str, tuple[float, float]]:
        """実験で設定してよい初期濃度の範囲。既定値 c の factor 倍。c = 0 の種は [0, 既定値の中央値]"""
        pos = [v for v in self.initial.values() if v > 0]
        scale = float(np.median(pos)) if pos else 1.0
        return {s: (factor[0] * self.initial[s], factor[1] * self.initial[s]) if self.initial.get(s, 0.0) > 0 else (0.0, scale)
                for s in self.perturbable}


_counter = {"n": 0}


@dataclass
class Candidate:
    """Antimony の反応ブロック 1 つ = 候補 SBML 1 つ。自由定数（仮説が足した大域パラメータ）は params に持つ"""

    block: str
    canonical: str
    sbml: str
    params: dict[str, float]
    system: System
    cid: str
    fit_error: float = math.inf

    def simulate_many(self, jobs: list[tuple[dict, dict]], index=None) -> list[np.ndarray | None]:  # noqa: ANN001
        return self.system.worker.run(self.cid, jobs, index)

    def simulate(self, initial: dict[str, float] | None = None, params: dict[str, float] | None = None) -> np.ndarray | None:
        return self.simulate_many([(params or self.params, initial or {})])[0]

    def sbml_with_params(self) -> str:
        doc = libsbml.readSBMLFromString(self.sbml)
        for pid, v in self.params.items():
            doc.getModel().getParameter(pid).setValue(float(v))
        return libsbml.writeSBMLToString(doc)


def make_candidate(block: str, system: System) -> Candidate:
    """ブロック → 正規化 → SBML → 子プロセスに読み込み → 既定条件で回ることを確認。駄目なら ValueError / RuntimeError"""
    canonical, normalized = canonicalize(block, system.species, system.compartments)
    sbml = to_sbml(system.base, normalized)
    m = libsbml.readSBMLFromString(sbml).getModel()
    params = {m.getParameter(i).getId(): m.getParameter(i).getValue() for i in range(m.getNumParameters())
              if m.getParameter(i).getId() not in system.existing_params}
    _counter["n"] += 1
    cand = Candidate(normalized, canonical, sbml, params, system, f"c{_counter['n']}")
    system.worker.load(cand.cid, sbml)
    if cand.simulate() is None:
        raise ValueError("does not simulate at default conditions")
    return cand


# ------------------------------------------------------------------ upstream subclasses
@dataclass
class TrajectoryResult(OracleResult):
    """OracleResult + 全種 × 全時刻の軌道（上流の measurement はスカラー）"""

    trajectory: np.ndarray | None = None


class TrajectoryStore(ExperimentStore):
    """ExperimentStore のまま。prompt 用の表だけ軌道の要約（5 時点 × 全種）にする"""

    def __init__(self, domain: str, parameter_names: list[str], species: list[str], n_points: int):
        super().__init__(domain, parameter_names)
        self.species = species
        self.show = np.unique(np.linspace(0, n_points - 1, 5).round().astype(int))

    def trajectories(self) -> list[np.ndarray]:
        return [r.trajectory for r in self._results]

    def summary(self) -> str:
        return f"N={len(self._results)} experiments, each a time series of {len(self.species)} species"

    def to_text_table(self, max_rows: int = 20) -> str:
        if not self._results:
            return "No experiments yet."
        lines = []
        for k, r in enumerate(self._results[-max_rows:], 1):
            pert = ", ".join(f"{p}={v:.4g}" for p, v in r.params.items()) or "default initial concentrations"
            lines.append(f"## Experiment {k}: {pert}")
            lines.append("species: " + " | ".join(f"t[{j}]" for j in self.show))
            for s_i, s in enumerate(self.species):
                lines.append(f"{s}: " + " | ".join(f"{r.trajectory[j, s_i]:.3g}" for j in self.show))
        return "\n".join(lines)


class SciGymOracle(BaseOracle):
    """BaseOracle の口で SciGym の Experiment.call_simulator を呼ぶ。候補の試走はここを通らないので予算を消費しない"""

    def __init__(self, system: System, experiment, bounds: dict[str, tuple[float, float]], instructions: str):  # noqa: ANN001
        self.system, self.experiment, self.bounds, self.instructions = system, experiment, bounds, instructions
        self.n_calls = 0

    @property
    def domain(self) -> str:
        return "scigym"

    @property
    def parameter_names(self) -> list[str]:
        return list(self.bounds)

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return self.bounds

    @property
    def function_signature(self) -> str:
        return "Antimony reaction block, e.g.\n  R1: id_a + id_b -> id_c; comp * k1 * id_a * id_b;\n  k1 = 0.5;"

    @property
    def param_description(self) -> str:
        return self.instructions

    def run(self, params: dict[str, float]) -> "TrajectoryResult":
        from scigym.api import ExperimentConfig

        self.n_calls += 1
        actions = [f'change_initial_concentration("{s}", {v})' for s, v in params.items()]
        res = self.experiment.call_simulator(ExperimentConfig(experiment_action=actions, observed_species=[]))
        if not res.success or res.result is None:
            raise RuntimeError(res.error_message)
        y = np.array([res.result[s] for s in self.system.species], dtype=float).T
        return TrajectoryResult(params=dict(params), measurement=float(np.nanmean(y)), domain="scigym", noise_level=0.0, trajectory=y)

    def evaluate_law(self, law_str: str):  # noqa: ANN201
        raise NotImplementedError


class TrajectoryDistribution(HypothesisDistribution):
    """HypothesisDistribution のまま（骨格・クラスタ・エントロピー）。compute_disagreement だけ軌道の候補間分散にする"""

    @classmethod
    def over(cls, members: list[tuple["Candidate", dict]], store: "TrajectoryStore", acquisition: str = "disagreement") -> "TrajectoryDistribution":
        # acquisition は run の設定（"disagreement" か "none"）を生のまま受ける。bool への変換はここでやり、yaml の値と観測が一致する
        dist = cls.from_expressions([c.canonical for c, _ in members])
        dist.members, dist.store, dist.enabled = members, store, acquisition == "disagreement"
        return dist

    def compute_disagreement(self, X: np.ndarray, param_names: list[str]) -> np.ndarray:
        n = X.shape[0]
        if not self.enabled or len(self.members) < 2:
            return np.zeros(n)  # アブレーション: スコア 0 → select は多様性だけで選ぶ
        system = self.members[0][0].system
        index = system.fit_index()
        eps = log_eps(self.store, system)
        logs = np.full((len(self.members), n, len(index), len(system.species)), np.nan)
        initials = [dict(zip(param_names, X[i].tolist())) for i in range(n)]
        for k, (cand, params) in enumerate(self.members):
            for i, y in enumerate(cand.simulate_many([(params, init) for init in initials], index)):
                if y is not None:
                    logs[k, i] = np.log(np.maximum(y, 0.0) + eps)
        var = np.nanvar(logs, axis=0)
        with np.errstate(all="ignore"):
            score = np.nanmean(var, axis=(1, 2))
        score[~np.isfinite(score)] = 0.0
        return score


# ------------------------------------------------------------------ fitting / confidence (own: upstream fits with PySR on scalars)
def log_eps(store: "TrajectoryStore", system: System) -> np.ndarray:
    """種ごとの log 変換の下駄: 観測の最大値の 1e-3"""
    tr = store.trajectories()
    top = np.max(np.stack(tr), axis=(0, 1)) if tr else np.array([system.initial.get(s, 0.0) for s in system.species])
    return 1e-3 * np.abs(top) + 1e-12


def log_targets(store: "TrajectoryStore", system: System, index: np.ndarray, which: list[int]) -> np.ndarray:
    eps = log_eps(store, system)
    tr = store.trajectories()
    return np.stack([np.log(tr[i][index] + eps) for i in which])


def log_prediction(cand: Candidate, store: "TrajectoryStore", index: np.ndarray, which: list[int], params: dict) -> np.ndarray:
    eps = log_eps(store, cand.system)
    out = np.full((len(which), len(index), len(cand.system.species)), np.nan)
    results = store.get_all()
    for k, y in enumerate(cand.simulate_many([(params, results[i].params) for i in which], index)):
        if y is not None:
            out[k] = np.log(np.maximum(y, 0.0) + eps)
    return out


def fit(cand: Candidate, store: "TrajectoryStore", which: list[int] | None = None, max_nfev: int = 150, restarts: int = 1,
        rng: np.random.Generator | None = None) -> float:
    """自由定数を log 空間で最小二乗フィットし、cand.params を更新して log RMSE を返す"""
    which = list(which if which is not None else range(len(store)))
    if not which or not cand.params:
        cand.fit_error = fit_error(cand, store, which) if which else math.inf
        return cand.fit_error
    index = cand.system.fit_index()
    target = log_targets(store, cand.system, index, which)
    ids = list(cand.params)
    lo, hi = np.full(len(ids), PARAM_LOG_BOUNDS[0]), np.full(len(ids), PARAM_LOG_BOUNDS[1])

    def residual(x: np.ndarray) -> np.ndarray:
        r = log_prediction(cand, store, index, which, dict(zip(ids, np.exp(x)))) - target
        r[~np.isfinite(r)] = FAIL_RESIDUAL
        return r.ravel()

    rng = rng or np.random.default_rng(0)
    starts = [np.log(np.maximum(np.array([cand.params[p] for p in ids], dtype=float), 1e-12))]
    starts += [rng.uniform(-6, 6, len(ids)) for _ in range(restarts)]
    best_x, best_cost = starts[0], math.inf
    for x0 in starts:
        try:
            sol = least_squares(residual, np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), max_nfev=max_nfev, method="trf")
        except Exception:
            continue
        if sol.cost < best_cost:
            best_x, best_cost = sol.x, sol.cost
    cand.params = dict(zip(ids, np.exp(best_x)))
    cand.fit_error = fit_error(cand, store, which)
    return cand.fit_error


def fit_error(cand: Candidate, store: "TrajectoryStore", which: list[int]) -> float:
    index = cand.system.fit_index()
    r = log_prediction(cand, store, index, which, cand.params) - log_targets(store, cand.system, index, which)
    r[~np.isfinite(r)] = FAIL_RESIDUAL
    return float(np.sqrt(np.mean(r ** 2)))


def log_r2(cand: Candidate, store: "TrajectoryStore") -> float:
    """log 濃度の決定係数（種ごとの平均を基準）"""
    index = cand.system.fit_index()
    rows = list(range(len(store)))
    P = log_prediction(cand, store, index, rows, cand.params)
    T = log_targets(store, cand.system, index, rows)
    P = np.where(np.isfinite(P), P, T + FAIL_RESIDUAL)
    ss_tot = float(np.sum((T - T.mean(axis=(0, 1), keepdims=True)) ** 2))
    return 1.0 - float(np.sum((P - T) ** 2)) / ss_tot if ss_tot > 0 else 0.0


def bootstrap_confidence(cand: Candidate, store: "TrajectoryStore", n_bootstrap: int, rng: np.random.Generator,
                         min_r2: float, max_nfev: int = 60) -> tuple[float, list[dict]]:
    """公式 learner.py の定義: 再標本化して再フィットし confidence = 1 − mean(std / |mean|)。R² が低い候補は 0（公式の門）"""
    if len(store) < 2 or log_r2(cand, store) < min_r2:
        return 0.0, []
    index = cand.system.fit_index()
    rows_all = list(range(len(store)))
    saved = dict(cand.params)
    refits, preds = [], []
    for _ in range(n_bootstrap):
        rows = rng.integers(0, len(store), len(store)).tolist()
        cand.params = dict(saved)
        fit(cand, store, which=rows, max_nfev=max_nfev, restarts=0)
        refits.append(dict(cand.params))
        preds.append(np.exp(log_prediction(cand, store, index, rows_all, cand.params)))
    cand.params = saved
    cand.fit_error = fit_error(cand, store, rows_all)
    P = np.stack(preds)
    if np.isnan(P).all():
        return 0.0, refits
    cv = np.nanstd(P, axis=0) / (np.abs(np.nanmean(P, axis=0)) + 1e-10)
    return float(np.clip(1.0 - float(np.nanmean(cv)), 0.0, 1.0)), refits
