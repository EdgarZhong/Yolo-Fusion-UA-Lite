import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import DualBackboneOBBModel, attempt_load_one_weight
from ultralytics.utils import RANK

RUN_NAME = "Exp-B_CoordAttn"
TOTAL_EPOCHS = 160
FINAL_CLOSE_EPOCHS = 16
FREEZE_EPOCHS = 10
USE_TEST_AS_VAL = True
PRETRAINED_WEIGHTS = ROOT / "yolov8n.pt"
SINGLE_TO_DUAL_BACKBONE_MAP = {
    0: (3, 13),
    1: (4, 14),
    2: (5, 15),
    3: (6, 16),
    4: (7, 17),
    5: (8, 18),
    6: (9, 19),
    7: (10, 20),
    8: (11, 21),
    9: (12, 22),
}


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
        "name": "attention_expb",
        "train_cmd": build_train_command(),
        "resume_cmd": build_resume_command(),
        "resume_ready": str(run_dir / "weights" / "last.pt"),
        "workdir": str(_root_dir()),
        "run_dir": str(run_dir),
        "total_epochs": TOTAL_EPOCHS,
    }


def transfer_single_backbone_to_dual(source_weights: Path, target_model) -> dict:
    if not source_weights.exists():
        raise FileNotFoundError(f"预训练权重不存在: {source_weights}")
    max_target_idx = max(max(v) for v in SINGLE_TO_DUAL_BACKBONE_MAP.values())
    if not hasattr(target_model, "model") or len(target_model.model) <= max_target_idx:
        raise ValueError(f"目标模型结构与映射不一致，最大目标层索引={max_target_idx}")

    source_model, _ = attempt_load_one_weight(str(source_weights))
    source_sd = source_model.float().state_dict()
    target_sd = target_model.state_dict()
    copied = 0
    skipped = 0
    source_hits = 0

    for source_idx, target_pair in SINGLE_TO_DUAL_BACKBONE_MAP.items():
        source_prefix = f"model.{source_idx}."
        for key, value in source_sd.items():
            if not key.startswith(source_prefix):
                continue
            source_hits += 1
            suffix = key[len(source_prefix):]
            for target_idx in target_pair:
                target_key = f"model.{target_idx}.{suffix}"
                if target_key in target_sd and target_sd[target_key].shape == value.shape:
                    target_sd[target_key].copy_(value)
                    copied += 1
                else:
                    skipped += 1

    if RANK in (-1, 0):
        print(f"[InitTransfer] source={source_weights}")
        print(f"[InitTransfer] copied={copied}, skipped={skipped}")

    if source_hits == 0 or copied == 0:
        raise RuntimeError(
            f"主干迁移失败: source_hits={source_hits}, copied={copied}，请检查映射与模型结构是否匹配"
        )

    return {"copied": copied, "skipped": skipped}


class ExpBOBBTrainer(OBBTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DualBackboneOBBModel(cfg, ch=6, nc=self.data["nc"], verbose=verbose and RANK == -1)
        transfer_single_backbone_to_dual(PRETRAINED_WEIGHTS, model)
        return model


def main():
    model_cfg = ROOT / "src" / "cfg" / "model" / "attn_exp_B_CoordAttn.yaml"
    data_cfg = ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml"

    run_dir = get_run_dir()
    project_dir = run_dir.parent
    name = run_dir.name

    overrides = {
        "task": "obb",
        "model": str(model_cfg),
        "data": str(data_cfg),
        "pretrained": False,
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

    trainer = ExpBOBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
