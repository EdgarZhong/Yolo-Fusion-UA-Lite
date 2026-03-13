import argparse
import csv
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_entry_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("train_entry", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载训练脚本：{script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_command(cmd):
    if cmd is None:
        return None
    if isinstance(cmd, (list, tuple)):
        return [str(x) for x in cmd]
    raise TypeError("训练脚本返回的命令必须为 list 或 tuple")


def normalize_spec(script_path: Path, module):
    spec = {}
    if hasattr(module, "get_train_manager_spec"):
        spec = module.get_train_manager_spec() or {}

    train_cmd = spec.get("train_cmd")
    if train_cmd is None and hasattr(module, "build_train_command"):
        train_cmd = module.build_train_command()
    if train_cmd is None:
        train_cmd = [sys.executable, str(script_path)]

    resume_cmd = spec.get("resume_cmd")
    if resume_cmd is None and hasattr(module, "build_resume_command"):
        resume_cmd = module.build_resume_command()

    resume_ready = spec.get("resume_ready")
    workdir = spec.get("workdir")
    name = spec.get("name") or script_path.stem
    run_dir = spec.get("run_dir")
    total_epochs = spec.get("total_epochs")

    if run_dir is None and resume_ready:
        resume_path = Path(resume_ready)
        if resume_path.name == "last.pt" and resume_path.parent.name == "weights":
            run_dir = resume_path.parent.parent
        else:
            run_dir = resume_path.parent

    return {
        "name": name,
        "train_cmd": normalize_command(train_cmd),
        "resume_cmd": normalize_command(resume_cmd),
        "resume_ready": Path(resume_ready) if resume_ready else None,
        "workdir": Path(workdir) if workdir else ROOT,
        "run_dir": Path(run_dir) if run_dir else None,
        "total_epochs": int(total_epochs) if total_epochs is not None else None,
    }


def read_last_epoch(results_csv: Path):
    if not results_csv.exists():
        return None
    try:
        with results_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None
            epoch_key = None
            for key in reader.fieldnames:
                if key.lower() == "epoch":
                    epoch_key = key
                    break
            if epoch_key is None:
                return None
            last_epoch = None
            for row in reader:
                value = row.get(epoch_key)
                if value is None or value == "":
                    continue
                last_epoch = int(float(value))
            return last_epoch
    except Exception:
        return None


def is_completed(run_dir: Path | None, total_epochs: int | None) -> bool:
    if run_dir is None or total_epochs is None:
        return False
    results_csv = run_dir / "results.csv"
    last_epoch = read_last_epoch(results_csv)
    if last_epoch is None:
        return False
    return (last_epoch + 1) >= total_epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, required=True)
    parser.add_argument("--max-retries", type=int, default=100)
    parser.add_argument("--cooldown", type=int, default=5)
    parser.add_argument("--check-interval", type=float, default=2.0)
    parser.add_argument("--launch-blocking", action="store_true")
    args = parser.parse_args()

    script_path = Path(args.script)
    if not script_path.is_file():
        raise FileNotFoundError(f"未找到训练脚本：{script_path}")

    module = load_entry_module(script_path)
    spec = normalize_spec(script_path, module)

    env = os.environ.copy()
    if args.launch_blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"

    attempt = 0
    while True:
        completed = is_completed(spec["run_dir"], spec["total_epochs"])
        if completed:
            print(f"[TrainManager] run_dir={spec['run_dir']} 已完成训练，退出")
            break

        resume_ready = spec["resume_ready"]
        can_resume = resume_ready is not None and resume_ready.exists() and spec["resume_cmd"] is not None

        use_resume = False
        if attempt == 0:
            use_resume = can_resume
        else:
            use_resume = can_resume

        cmd = spec["resume_cmd"] if use_resume else spec["train_cmd"]

        print(f"[TrainManager] name={spec['name']}, attempt={attempt}, mode={'resume' if use_resume else 'train'}")
        print(f"[TrainManager] cmd={' '.join(cmd)}")
        proc = subprocess.Popen(cmd, cwd=str(spec["workdir"]), env=env)

        exit_code = None
        while exit_code is None:
            exit_code = proc.poll()
            if exit_code is None:
                time.sleep(max(0.1, float(args.check_interval)))

        if exit_code == 0:
            print("[TrainManager] 训练进程正常结束")
            break

        print(f"[TrainManager] 训练进程异常退出，code={exit_code}")
        attempt += 1
        if attempt > args.max_retries:
            print("[TrainManager] 达到最大重试次数，结束")
            break
        time.sleep(max(1, int(args.cooldown)))


if __name__ == "__main__":
    main()
