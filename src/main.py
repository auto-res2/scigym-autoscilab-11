"""1 つの run（= 1 手法 × 1 split）で SciGym-small の各件を解かせ、評価層 scigym_<split> の入力ファイルを書く。

method: react     → src.train（SciGym 公式 Controller）
method: autoscilab → src.inference（LLM-AutoSciLab の部品 + SciGym 用アダプタ）
件ごとに 1 プロセス（src/runner.py）。モデルは Vercel AI Gateway 経由。
run の設定は config/run/<run_id>.yaml にあり、キーは値を消費する関数の引数名（src/keys.py）。
"""

import gzip
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import yaml

from src import keys as K
from src.preprocess import instances
from src.runner import BUDGET_EXCEEDED, run_all


def cli_args():
    """hydra 形式の key=value を読む。hydra / omegaconf は scigym が固定する petab の antlr 版と衝突する"""
    return {k: yaml.safe_load(v) for k, _, v in (a.partition("=") for a in sys.argv[1:])}


def self_checks(cfg) -> dict[str, bool]:
    """凍結の条件（sanity で確認する）。速度則の型の列挙が無い、候補の試走が予算を消費しない、Refine のスコアが再フィット間、同構造が同クラスタ"""
    src = "".join(Path(p).read_text() for p in Path("src").glob("*.py") if p.name != "main.py")
    forbidden = re.compile("|".join(["mass_" + "action", "michaelis_" + "menten", r"\bhill\b"]))  # 自分自身にマッチしないよう分割
    checks = {"no_rate_law_enum_in_src": forbidden.search(src) is None}
    if cfg.method != "autoscilab":
        return checks
    import copy

    import numpy as np
    from scigym.actions.experiment import Experiment
    from scigym.data.question import Question

    from src.model import SciGymOracle, System, TrajectoryDistribution, TrajectoryStore, make_candidate
    inst = instances(cfg.data_root, cfg.run[K.SPLIT], "sanity", None)[0]
    q = Question(sbml_directory_path=str(inst), task_difficulty="fully_observable")
    system = System.from_incomplete(q.get_partial_sbml(), (inst / "truth.sedml").read_text())
    sp, comp = system.species, system.compartments[0]
    a = make_candidate(f"{sp[0]} -> {sp[1]}; {comp}*k*{sp[0]}^2/(K^2+{sp[0]}^2); k = 1; K = 0.5;", system)  # 自由な速度則が通る
    b = make_candidate(f"R: {sp[0]} -> {sp[1]}; {comp}*q*{sp[0]}^2/(Q^2+{sp[0]}^2); Q = 7; q = 3;", system)  # 同じ構造、定数と順序だけ違う
    checks["free_rate_law_round_trip"] = a.simulate() is not None
    checks["same_structure_same_cluster"] = len(TrajectoryDistribution.from_expressions([a.canonical, b.canonical]).clusters) == 1
    exp = Experiment(true_sbml=copy.deepcopy(q.get_original_sbml()), inco_sbml=copy.deepcopy(q.get_partial_sbml()))
    oracle = SciGymOracle(system, exp, system.allowed_bounds((0.2, 5.0)), "")
    store = TrajectoryStore("scigym", list(oracle.bounds), sp, system.n_points)
    store.add(oracle.run({}))
    X = np.array([[np.mean(v) for v in oracle.bounds.values()]] * 3)
    TrajectoryDistribution.over([(a, a.params), (b, b.params)], store).compute_disagreement(X, oracle.parameter_names)
    checks["candidate_simulation_consumes_no_budget"] = oracle.n_calls == 1
    checks["refine_scores_bootstrap_refits"] = TrajectoryDistribution.over([(a, a.params), (a, {k: v * 2 for k, v in a.params.items()})], store).compute_disagreement(X, oracle.parameter_names).max() > 0
    checks["ablation_scores_zero"] = TrajectoryDistribution.over([(a, a.params), (b, b.params)], store, acquisition="none").compute_disagreement(X, oracle.parameter_names).max() == 0
    system.worker.close()
    return {k: bool(v) for k, v in checks.items()}  # numpy の bool は JSON にならない


def main():
    cli = cli_args()
    run_id = cli["run"]
    cfg = yaml.safe_load(open("config/config.yaml"))
    cfg.update(cli)
    cfg["run"] = run = yaml.safe_load(open(f"config/run/{run_id}.yaml"))  # CLI の run=<run_id> を yaml の中身で置き換える
    cfg = SimpleNamespace(**cfg, method=run[K.METHOD], task=f"scigym_{run[K.SPLIT]}")
    run_dir = Path(cfg.results_dir) / run_id
    todo = instances(cfg.data_root, run[K.SPLIT], cfg.mode, run.get(K.N_INSTANCES))
    stage = cfg.mode.upper()
    checks = self_checks(cfg) if cfg.mode == "sanity" else {}
    if not all(checks.values()):
        print(f"{stage}_VALIDATION: FAIL reason=self_check:{','.join(k for k, v in checks.items() if not v)}")
        sys.exit(1)
    # 打ち切った前 run の件ごとの書庫（リポジトリの resume/<run_id>/）から続きを実行する
    if not (run_dir / "instances").exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        for archive in sorted(Path("resume", run_id).glob("**/*.tar.gz")):
            with tarfile.open(archive) as tar:
                tar.extractall(run_dir)
    # API エラーで tokens.json が出なかった件は 2 回までやり直す。予算超過（402）は即座に run を止める
    for _ in range(3):
        codes = run_all(todo, run_dir, cfg, run[K.METHOD], run[K.WORKERS], run[K.INSTANCE_TIMEOUT], run[K.STALL_TIMEOUT])
        if BUDGET_EXCEEDED in codes:
            print(f"{stage}_VALIDATION: FAIL reason=budget_exceeded")
            sys.exit(1)
    submitted, totals = {}, {"input_tokens": 0, "output_tokens": 0}
    n_experiments, n_overflow = [], 0
    for instance in todo:
        out = run_dir / "instances" / instance.name
        if (out / "context_overflow").exists():
            n_overflow += 1
        if not (out / "tokens.json").exists():
            continue  # 文脈超過か、3 試行とも終わらなかった件: 提出無しとして評価層に渡す
        info = json.loads((out / "tokens.json").read_text())
        submitted[instance.name] = (out / "final_model.xml").read_text() if (out / "final_model.xml").exists() else None
        n_experiments.append(info["n_experiments"])
        for k in totals:
            totals[k] += info[k]
    totals |= {
        "mean_experiments": sum(n_experiments) / len(n_experiments) if n_experiments else 0.0,
        "n_with_submission": sum(v is not None for v in submitted.values()),
        "n_context_overflow": n_overflow,
        "n_not_finished": len(todo) - len(submitted),
    }
    (run_dir / "eval_inputs").mkdir(parents=True, exist_ok=True)
    # 評価入力は gzip で書く（参照 SBML と提出 SBML を全件持つので大きい）
    with gzip.open(run_dir / "eval_inputs" / f"{cfg.task}.json.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"instances": [{
            "id": p.name,
            "reference_sbml": (p / "truth.xml").read_text(),
            "incomplete_sbml": (p / "partial.xml").read_text(),
            "reference_sedml": (p / "truth.sedml").read_text(),
            "submitted_sbml": submitted.get(p.name),
        } for p in todo]}))
    (run_dir / "tokens.json").write_text(json.dumps(totals))
    # 取り込みは 1 ファイル 1 API 呼び出しなので、件ごとの出力は 1 つの書庫にまとめる（GitHub の secondary rate limit 対策）
    with tarfile.open(run_dir / "instances.tar.gz", "w:gz") as tar:
        tar.add(run_dir / "instances", arcname="instances")
    shutil.rmtree(run_dir / "instances")
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)  # 全件そろったので件ごとの書庫は不要
    print(f"{stage}_VALIDATION_SUMMARY: {json.dumps({'n_instances': len(todo), **totals, **checks})}")
    if totals["n_with_submission"] == 0:
        print(f"{stage}_VALIDATION: FAIL reason=no_submission")
        sys.exit(1)
    print(f"{stage}_VALIDATION: PASS")


if __name__ == "__main__":
    main()
