import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBTrainer

RUN_NAME = "Exp-A-P3-Inception-Concat"
TOTAL_EPOCHS = 160
FINAL_CLOSE_EPOCHS = 16
FREEZE_EPOCHS = 10
USE_TEST_AS_VAL = True


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def get_run_dir() -> Path:
    root = _root_dir()
    project_dir = root / "models" / "attention_exp"
    return project_dir / RUN_NAME


def build_train_command() -> list[str]:
    return [sys.executable, str(Path(__file__).resolve())]


def build_resume_command() -> list[str]:
    root = _root_dir()
    run_dir = get_run_dir()
    return [
        sys.executable,
        str(root / "src" / "trainning" / "resume_train.py"),
        "--resume",
        str(run_dir),
    ]


def get_train_manager_spec() -> dict:
    run_dir = get_run_dir()
    return {
        "name": "attention_expa",
        "train_cmd": build_train_command(),
        "resume_cmd": build_resume_command(),
        "resume_ready": str(run_dir / "weights" / "last.pt"),
        "workdir": str(_root_dir()),
        "run_dir": str(run_dir),
        "total_epochs": TOTAL_EPOCHS,
    }


def main():
    model_cfg = ROOT / "src" / "cfg" / "model" / "Exp-A_P3-Inception-Concat_P45-Concat.yaml"
    data_cfg = ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml"
    pretrained_weights = ROOT / "yolov8n.pt"

    run_dir = get_run_dir()
    project_dir = run_dir.parent
    name = run_dir.name

    overrides = {
        "task": "obb",
        "model": str(model_cfg),
        "data": str(data_cfg),
        "pretrained": str(pretrained_weights),
        "epochs": TOTAL_EPOCHS,
        "batch": 16,
        "imgsz": 640,
        "device": 0,
        "workers": 2,
        "patience": 0,
        "freeze": FREEZE_EPOCHS,
        "rect": False,
        "conf": 0.25,
        "iou": 0.75,
        "max_det": 500,
        "mosaic": 1.0,
        "close_mosaic": FINAL_CLOSE_EPOCHS,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "erasing": 0.0,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "use_test_as_val": USE_TEST_AS_VAL,
        "drop_prob_rgb": 0.10,
        "drop_prob_ir": 0.10,
        "close_dropout": FINAL_CLOSE_EPOCHS,
        "project": str(project_dir),
        "name": name,
        "exist_ok": True,
        "save": True,
        "val": True,
        "plots": True,
        "deterministic": True,
        "seed": 0,
    }

    trainer = OBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
