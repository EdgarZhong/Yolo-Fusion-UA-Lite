"""
RGB 单模态 SimAM 基准训练脚本（COCO 主干迁移初始化）

设计目标：
- 保持单主干 OBB 训练流程不变；
- 仅在 P3 侧路插入单模态版本的 Inception + SimAM 模块；
- 训练超参与注意力对比实验 Exp-0/A/B/C 保持一致，保证公平对照。
"""

import sys
from pathlib import Path

# ================== 仓库根路径与 Ultralytics 源码导入 ==================
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import OBBModel, attempt_load_one_weight
from ultralytics.utils import RANK

# ================== 训练实验固定配置 ==================
RUN_NAME = "RGB-Only-SimAM"
TOTAL_EPOCHS = 160
CLOSE_FINAL_EPOCHS = 16
FREEZE_EPOCHS = 10
USE_TEST_AS_VAL = True
PRETRAINED_WEIGHTS = ROOT / "yolov8n.pt"
MODEL_CFG = ROOT / "src" / "cfg" / "model" / "single_modal_p3_inception_simam.yaml"
DATA_YAML = ROOT / "src" / "cfg" / "datasets" / "rgb_obb_dronevehicle.yaml"
SINGLE_TO_RGB_BACKBONE_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
}


def _root_dir() -> Path:
    """返回仓库根目录。"""
    return Path(__file__).resolve().parents[2]


def get_run_dir() -> Path:
    """返回本次训练的输出目录。"""
    root = _root_dir()
    project_dir = root / "models" / "baseline"
    return project_dir / RUN_NAME


def build_train_command() -> list[str]:
    """构建 Train Manager 启动训练时使用的命令。"""
    return [sys.executable, str(Path(__file__).resolve())]


def build_resume_command() -> list[str]:
    """构建 Train Manager 断点续训时使用的命令。"""
    root = _root_dir()
    run_dir = get_run_dir()
    return [
        sys.executable,
        str(root / "src" / "trainning" / "resume_train.py"),
        "--resume",
        str(run_dir),
    ]


def get_train_manager_spec() -> dict:
    """输出 Train Manager 所需的规范字段。"""
    run_dir = get_run_dir()
    return {
        "name": "rgb_only_simam_pretrained_baseline",
        "train_cmd": build_train_command(),
        "resume_cmd": build_resume_command(),
        "resume_ready": str(run_dir / "weights" / "last.pt"),
        "workdir": str(_root_dir()),
        "run_dir": str(run_dir),
        "total_epochs": TOTAL_EPOCHS,
    }


def transfer_single_backbone_to_rgb_simam(source_weights: Path, target_model) -> dict:
    """把标准 yolov8n 主干参数迁移到插入单模态 SimAM 后的新模型中。"""
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

    # 逐层复制 backbone 参数，确保新增的 SimAM 模块保持独立初始化
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
        print(f"[InitTransfer][RGB-SimAM] source={source_weights}")
        print(f"[InitTransfer][RGB-SimAM] copied={copied}, skipped={skipped}")

    if source_hits == 0 or copied == 0:
        raise RuntimeError(
            f"主干迁移失败: source_hits={source_hits}, copied={copied}，请检查映射与模型结构是否匹配"
        )

    return {"copied": copied, "skipped": skipped}


class RGBSimAMOBBTrainer(OBBTrainer):
    """RGB 单模态 SimAM 训练器。"""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """构建单模态 SimAM 模型并执行主干迁移。"""
        model = OBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose)
        transfer_single_backbone_to_rgb_simam(PRETRAINED_WEIGHTS, model)
        return model


def main() -> None:
    """训练主入口。"""
    if not MODEL_CFG.exists():
        raise FileNotFoundError(f"未找到模型配置: {MODEL_CFG}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"未找到数据配置: {DATA_YAML}")

    run_dir = get_run_dir()
    project_dir = run_dir.parent
    name = run_dir.name

    # 注意：以下超参刻意与注意力对比实验保持一致，
    # 即便单模态下 drop_prob_* 不生效，也显式写出，防止口径漂移。
    overrides = {
        "task": "obb",
        "model": str(MODEL_CFG),
        "pretrained": False,
        "data": str(DATA_YAML),
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
        "close_mosaic": CLOSE_FINAL_EPOCHS,
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
        "close_dropout": CLOSE_FINAL_EPOCHS,
        "project": str(project_dir),
        "name": name,
        "exist_ok": True,
        "save": True,
        "val": True,
        "plots": True,
        "deterministic": True,
        "seed": 0,
    }

    print(
        f"[Train][RGB-SimAM] 启动训练：epochs={overrides['epochs']}, batch={overrides['batch']}, imgsz={overrides['imgsz']}"
    )
    print(f"[Train][RGB-SimAM] 模型配置: {MODEL_CFG}")
    print(f"[Train][RGB-SimAM] 数据配置: {DATA_YAML}")
    print(f"[Train][RGB-SimAM] 主干迁移权重: {PRETRAINED_WEIGHTS}")
    print(f"[Train][RGB-SimAM] 输出目录: {run_dir}")

    trainer = RGBSimAMOBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
