import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from src.trainning.train_manager import load_entry_module, normalize_spec
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import DualBackboneOBBModel, OBBModel


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _check_manager_contract(script_path: Path, module) -> dict:
    if not hasattr(module, "get_train_manager_spec"):
        _fail("训练脚本未实现 get_train_manager_spec()")
    spec = normalize_spec(script_path, module)
    missing = []
    for key in ("train_cmd", "resume_cmd", "resume_ready", "run_dir", "total_epochs"):
        if spec.get(key) in (None, "", []):
            missing.append(key)
    if missing:
        _fail(f"TrainManager 规范字段缺失: {missing}")
    return spec


def _check_transfer_contract(module) -> None:
    has_transfer = hasattr(module, "transfer_single_backbone_to_dual")
    if not has_transfer:
        return
    if not hasattr(module, "PRETRAINED_WEIGHTS"):
        _fail("存在迁移函数但缺少 PRETRAINED_WEIGHTS")
    pw = Path(getattr(module, "PRETRAINED_WEIGHTS"))
    if not pw.exists():
        _fail(f"PRETRAINED_WEIGHTS 不存在: {pw}")


def _check_framework_freeze_semantics() -> None:
    single_cfg = ROOT / "ultralytics-8.2" / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-obb.yaml"
    dual_cfg = ROOT / "src" / "cfg" / "model" / "Exp-0_P3-FA-Concat_P45-Concat.yaml"
    single_model = OBBModel(cfg=str(single_cfg), ch=3, nc=5, verbose=False)
    dual_model = DualBackboneOBBModel(cfg=str(dual_cfg), ch=6, nc=5, verbose=False)

    d_single = SimpleNamespace(args=SimpleNamespace(freeze=10), model=single_model)
    d_dual = SimpleNamespace(args=SimpleNamespace(freeze=10), model=dual_model)
    single_list, single_epoch = BaseTrainer._resolve_freeze_plan(d_single)
    dual_list, dual_epoch = BaseTrainer._resolve_freeze_plan(d_dual)

    if single_epoch != 10 or dual_epoch != 10:
        _fail("freeze 轮语义异常：auto_unfreeze_epoch 不为 10")
    if not single_list:
        _fail("单主干 freeze 计划为空")
    if not dual_list:
        _fail("双主干 freeze 计划为空")
    if min(dual_list) > 3 or max(dual_list) < 22:
        _fail("双主干 freeze 计划未覆盖预期 backbone 范围")


def _run_dynamic_check(spec: dict, timeout_sec: int) -> None:
    resume_ready = spec["resume_ready"]
    if resume_ready is None or not Path(resume_ready).exists():
        _fail("动态回归需要可用的 resume_ready(last.pt)，当前不存在")
    cmd = spec["resume_cmd"]
    if cmd is None:
        _fail("动态回归需要 resume_cmd，当前为空")

    proc = subprocess.Popen(
        cmd,
        cwd=str(spec["workdir"]),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines = []
    start = time.time()
    try:
        while time.time() - start < timeout_sec:
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                lines.append(line.rstrip("\n"))
            if proc.poll() is not None:
                break
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
    finally:
        if proc.poll() is None:
            proc.kill()

    log_text = "\n".join(lines)
    if "Freeze plan:" not in log_text:
        _fail("动态回归失败：未检测到 Freeze plan 日志")
    has_backbone_freeze_log = any(
        ("Freezing layer 'model." in line) and (".dfl." not in line) for line in lines
    )
    if "skip backbone freezing" in log_text:
        if has_backbone_freeze_log:
            _fail("动态回归失败：已跳过冻结但仍出现 backbone 冻结日志")
    else:
        if not has_backbone_freeze_log:
            _fail("动态回归失败：未跳过冻结且也未观察到 backbone 冻结日志")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, required=True)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=90)
    args = parser.parse_args()

    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = (ROOT / script_path).resolve()
    if not script_path.exists():
        _fail(f"训练脚本不存在: {script_path}")

    module = load_entry_module(script_path)
    spec = _check_manager_contract(script_path, module)
    _check_transfer_contract(module)
    _check_framework_freeze_semantics()
    if args.dynamic:
        _run_dynamic_check(spec, args.timeout_sec)

    print("[RegressionGate] PASS")
    print(f"[RegressionGate] script={script_path}")
    print(f"[RegressionGate] dynamic={args.dynamic}")


if __name__ == "__main__":
    main()
