"""
RGB 单模态 YOLOv8n OBB 检测模型训练脚本（COCO 主干迁移初始化）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import OBBModel, attempt_load_one_weight
from ultralytics.utils import RANK, yaml_load, yaml_save

RUN_NAME = "RGB-Only-Pretrained-New"
TOTAL_EPOCHS = 160
CLOSE_MOSAIC_EPOCHS = 16
FREEZE_EPOCHS = 10
PRETRAINED_WEIGHTS = ROOT / "yolov8n.pt"
DUAL_DATA_YAML = ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml"
IMG_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}
SINGLE_TO_RGB_BACKBONE_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
}


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def get_run_dir() -> Path:
    root = _root_dir()
    project_dir = root / "models" / "baseline"
    return project_dir / RUN_NAME


def _collect_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.rglob("*.*") if p.suffix.lower() in IMG_EXTS)


def _write_manifest(image_dir: Path, manifest_path: Path) -> int:
    if not image_dir.exists():
        raise FileNotFoundError(f"RGB 图像目录不存在: {image_dir}")
    files = _collect_images(image_dir)
    if not files:
        raise RuntimeError(f"RGB 图像目录为空: {image_dir}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(p.as_posix() for p in files), encoding="utf-8")
    return len(files)


def build_rgb_dataset_yaml() -> Path:
    if not DUAL_DATA_YAML.exists():
        raise FileNotFoundError(f"未找到双模态数据配置: {DUAL_DATA_YAML}")

    data_cfg = yaml_load(DUAL_DATA_YAML)
    runtime_dir = get_run_dir() / "_runtime_data"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    train_dir = (ROOT / data_cfg["train"]).resolve()
    val_dir = (ROOT / data_cfg["val"]).resolve()
    test_dir = (ROOT / data_cfg["test"]).resolve()

    train_list = runtime_dir / "train_rgb.txt"
    val_list = runtime_dir / "val_rgb.txt"
    test_list = runtime_dir / "test_rgb.txt"

    train_count = _write_manifest(train_dir, train_list)
    val_count = _write_manifest(val_dir, val_list)
    test_count = _write_manifest(test_dir, test_list)

    runtime_yaml = runtime_dir / "rgb_obb_dronevehicle_runtime.yaml"
    runtime_cfg = {
        "path": str(ROOT),
        "train": train_list.as_posix(),
        "val": val_list.as_posix(),
        "test": test_list.as_posix(),
        "nc": data_cfg["nc"],
        "names": data_cfg["names"],
        "task": "obb",
    }
    yaml_save(runtime_yaml, runtime_cfg)

    if RANK in (-1, 0):
        print(f"[Data][RGB-OBB] train={train_count}, val={val_count}, test={test_count}")
        print(f"[Data][RGB-OBB] runtime_yaml={runtime_yaml}")

    return runtime_yaml


def transfer_single_backbone_to_rgb_obb(source_weights: Path, target_model) -> dict:
    if not source_weights.exists():
        raise FileNotFoundError(f"预训练权重不存在: {source_weights}")
    max_target_idx = max(SINGLE_TO_RGB_BACKBONE_MAP.values())
    if not hasattr(target_model, "model") or len(target_model.model) <= max_target_idx:
        raise ValueError(f"目标模型结构与映射不一致，最大目标层索引={max_target_idx}")

    source_model, _ = attempt_load_one_weight(str(source_weights))
    source_sd = source_model.float().state_dict()
    target_sd = target_model.state_dict()
    copied = 0
    skipped = 0
    source_hits = 0

    for source_idx, target_idx in SINGLE_TO_RGB_BACKBONE_MAP.items():
        source_prefix = f"model.{source_idx}."
        for key, value in source_sd.items():
            if not key.startswith(source_prefix):
                continue
            source_hits += 1
            suffix = key[len(source_prefix):]
            target_key = f"model.{target_idx}.{suffix}"
            if target_key in target_sd and target_sd[target_key].shape == value.shape:
                target_sd[target_key].copy_(value)
                copied += 1
            else:
                skipped += 1

    if RANK in (-1, 0):
        print(f"[InitTransfer][RGB] source={source_weights}")
        print(f"[InitTransfer][RGB] copied={copied}, skipped={skipped}")

    if source_hits == 0 or copied == 0:
        raise RuntimeError(
            f"主干迁移失败: source_hits={source_hits}, copied={copied}，请检查映射与模型结构是否匹配"
        )

    return {"copied": copied, "skipped": skipped}


class RGB_OBB_Trainer(OBBTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = OBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose)
        transfer_single_backbone_to_rgb_obb(PRETRAINED_WEIGHTS, model)
        return model


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
        "name": "rgb_only_pretrained_baseline",
        "train_cmd": build_train_command(),
        "resume_cmd": build_resume_command(),
        "resume_ready": str(run_dir / "weights" / "last.pt"),
        "workdir": str(_root_dir()),
        "run_dir": str(run_dir),
        "total_epochs": TOTAL_EPOCHS,
    }


def main():
    model_cfg = ROOT / "ultralytics-8.2" / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-obb.yaml"
    data_yaml = build_rgb_dataset_yaml()
    run_dir = get_run_dir()
    project_dir = run_dir.parent
    name = run_dir.name

    overrides = {
        "task": "obb",
        "model": str(model_cfg),
        "pretrained": False,
        "data": str(data_yaml),
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
        "max_det": 300,
        "mosaic": 1.0,
        "close_mosaic": CLOSE_MOSAIC_EPOCHS,
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
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "use_test_as_val": True,
        "project": str(project_dir),
        "name": name,
        "exist_ok": True,
        "save": True,
        "val": True,
        "plots": True,
        "deterministic": True,
        "seed": 0,
    }

    print(f"[Train][RGB-OBB] 启动训练：epochs={overrides['epochs']}, batch={overrides['batch']}, imgsz={overrides['imgsz']}")
    print(f"[Train][RGB-OBB] 数据配置: {data_yaml}")
    print(f"[Train][RGB-OBB] 主干迁移权重: {PRETRAINED_WEIGHTS}")
    print(f"[Train][RGB-OBB] 输出目录: {run_dir}")

    trainer = RGB_OBB_Trainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
