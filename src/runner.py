"""件ごとに executor を 1 プロセス起動し、打ち切りとやり直しを管理する。

main.py ではなくここに置くのは、エントリモジュールの関数は観測で __main__.* になり、
run の設定（method、workers、timeout）を宣言どおりの名前で記録できないため。
"""

import json
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src import keys as K

BUDGET_EXCEEDED = 42
CONTEXT_OVERFLOW = 43
EXECUTOR = {"react": "src.train", "autoscilab": "src.inference"}
SANITY_REACT_ITERATIONS = 2
SANITY_AUTOSCILAB_BUDGET = 9  # 観測 1 + 4 + 4: 2 反復回って確信度ゲートまで通る最小


def instance_args(cfg, instance, out):
    """executor に渡す設定: run の yaml 全体 + 件の場所。sanity だけは予算を縮める"""
    args = {"instance_dir": str(instance), "out_dir": str(out), **cfg.run}
    if cfg.mode == "sanity":
        if cfg.method == "react":
            args[K.REACT_MAX_ITERATIONS] = SANITY_REACT_ITERATIONS
        else:
            args[K.TOTAL_BUDGET] = SANITY_AUTOSCILAB_BUDGET
    return args


def run_instance(cfg, run_dir, instance, method, instance_timeout, stall_timeout):
    out = run_dir / "instances" / instance.name
    if (out / "tokens.json").exists():
        return 0
    if (out / "context_overflow").exists():
        return 0  # 文脈長を超えた件。やり直しても同じなので「提出なし」として不完全モデルで採点する
    if (out / "stdout.txt").exists() and (out / "stdout.txt").read_text().count("killed after") >= 3:
        return 0  # 3 試行とも上限で打ち切られた件。公式の「有効な提出なし」と同じく不完全モデルで採点する
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "stdout.txt"
    with open(log_path, "a") as log:
        proc = subprocess.Popen([sys.executable, "-m", EXECUTOR[method], json.dumps(instance_args(cfg, instance, out))],
                                stdout=log, stderr=subprocess.STDOUT)
        started = time.time()
        while proc.poll() is None:
            time.sleep(30)
            # LLM のコードや提出モデルの評価が ODE の C ライブラリ内で止まると公式の 3 分制限（SIGALRM）が効かない。
            # 両方の executor は反復ごとに出力するので、出力が止まったままの試行は打ち切って最初からやり直す
            silent = time.time() - log_path.stat().st_mtime
            if time.time() - started > instance_timeout or silent > stall_timeout:
                proc.kill()
                proc.wait()
                print(f"[{instance.name}] killed after {int(time.time() - started)}s ({int(silent)}s without output)", file=log)
                return 1
    if proc.returncode == CONTEXT_OVERFLOW:
        (out / "context_overflow").touch()
    if not (out / "tokens.json").exists():  # 失敗した run の作業ディレクトリは残らないので原因を標準出力へ
        print(f"[{instance.name}] no tokens.json; log tail:", *(out / "stdout.txt").read_text().splitlines()[-25:], sep="\n  ")
    if (out / "tokens.json").exists() or (out / "context_overflow").exists():
        checkpoint(run_dir, instance.name)
    return proc.returncode


def run_all(todo: list[Path], run_dir: Path, cfg, method: str, workers: int, instance_timeout: float, stall_timeout: float) -> list[int]:
    """全件を並列に走らせて終了コードを返す。run の設定は生の値で受ける（この引数が観測され、yaml と比べられる）"""
    with ThreadPoolExecutor(int(workers)) as pool:
        return list(pool.map(lambda p: run_instance(cfg, run_dir, p, method, instance_timeout, stall_timeout), todo))


def checkpoint(run_dir, name):
    """終わった件を 1 件 1 書庫で残す。job が上限で打ち切られても、集めた書庫から続きを実行できる"""
    out = run_dir / "instances" / name
    shutil.rmtree(out / "codes", ignore_errors=True)  # 反復ごとのコードは chat_history.yaml に含まれる
    (out / "chat_history_readable.txt").unlink(missing_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with tarfile.open(run_dir / "checkpoints" / f"{name}.tar.gz", "w:gz") as tar:
        tar.add(out, arcname=f"instances/{name}")
