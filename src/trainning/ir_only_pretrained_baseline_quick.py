"""
IR 单模态 YOLOv8n OBB 快速验证脚本（COCO 主干迁移）

使用标准 YOLOv8n OBB 架构（3 通道输入），COCO 主干迁移初始化。
通过自定义 IR_OBB_Trainer 重写 get_model() 避免使用 DualBackboneOBBModel。

验证通过后，使用 ir_only_pretrained_baseline.py 进行全量训练。
"""

import sys
from pathlib import Path

# ================== 仓库根路径与 Ultralytics 源码导入 ==================
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

# 导入 OBB 训练器与模型构建工具
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import OBBModel, attempt_load_one_weight
from ultralytics.utils import RANK

PRETRAINED_WEIGHTS = ROOT / "yolov8n.pt"
RUN_NAME = "IR-Only-Pretrained-Quick"
SINGLE_TO_IR_BACKBONE_MAP = {
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


def transfer_single_backbone_to_ir_obb(source_weights: Path, target_model) -> dict:
    if not source_weights.exists():
        raise FileNotFoundError(f"预训练权重不存在: {source_weights}")
    max_target_idx = max(SINGLE_TO_IR_BACKBONE_MAP.values())
    if not hasattr(target_model, "model") or len(target_model.model) <= max_target_idx:
        raise ValueError(f"目标模型结构与映射不一致，最大目标层索引={max_target_idx}")

    source_model, _ = attempt_load_one_weight(str(source_weights))
    source_sd = source_model.float().state_dict()
    target_sd = target_model.state_dict()
    copied = 0
    skipped = 0
    source_hits = 0

    for source_idx, target_idx in SINGLE_TO_IR_BACKBONE_MAP.items():
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
        print(f"[InitTransfer][IR-Quick] source={source_weights}")
        print(f"[InitTransfer][IR-Quick] copied={copied}, skipped={skipped}")

    if source_hits == 0 or copied == 0:
        raise RuntimeError(
            f"主干迁移失败: source_hits={source_hits}, copied={copied}，请检查映射与模型结构是否匹配"
        )

    return {"copied": copied, "skipped": skipped}


class IR_OBB_Trainer(OBBTrainer):
    """
    IR 单模态 OBB 训练器：
    - 重写 get_model，构建 3 通道输入的标准 YOLOv8n OBB 模型
    - 加载 COCO 预训练权重 yolov8n.pt 并执行主干迁移初始化
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        构建 3 通道标准 YOLOv8n OBB 模型，并加载预训练权重
        """
        # 使用标准 OBBModel，3 通道输入
        model = OBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose)
        transfer_single_backbone_to_ir_obb(PRETRAINED_WEIGHTS, model)
        return model


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def get_run_dir() -> Path:
    root = _root_dir()
    project_dir = root / "models" / "baseline"
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
        "name": "ir_only_pretrained_baseline_quick",
        "train_cmd": build_train_command(),
        "resume_cmd": build_resume_command(),
        "resume_ready": str(run_dir / "weights" / "last.pt"),
        "workdir": str(_root_dir()),
        "run_dir": str(run_dir),
        "total_epochs": 1,
    }


def main():
    """
    快速验证训练流程（1 epoch，10% 数据，使用测试集作为验证集）
    """
    # -------------------------------------------------------------------------
    # 配置路径与参数
    # -------------------------------------------------------------------------
    MODEL_CFG = ROOT / "ultralytics-8.2" / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-obb.yaml"

    # 数据集配置（IR 单模态，指向 data_croped/ 目录）
    DATA_YAML = ROOT / "src" / "cfg" / "datasets" / "ir_obb_dronevehicle.yaml"

    # 快速验证输出目录
    RUN_DIR = get_run_dir()
    PROJECT_DIR = RUN_DIR.parent
    NAME = RUN_DIR.name

    # -------------------------------------------------------------------------
    # 构建训练参数覆盖字典
    # -------------------------------------------------------------------------
    overrides = {
        # 任务与核心路径
        "task": "obb",
        "model": str(MODEL_CFG),
        "pretrained": False,
        "data": str(DATA_YAML),
        # 快速验证规模（1 epoch，10% 数据）
        "epochs": 1,
        "batch": 16,
        "imgsz": 640,
        "device": 0,
        "workers": 0,                       # Windows 下使用 0 避免多进程问题
        "fraction": 0.1,                    # 仅使用 10% 数据子集
        # 训练策略（快速验证不冻结）
        "patience": 0,
        "freeze": 0,
        "rect": False,
        # 验证参数
        "conf": 0.25,
        "iou": 0.75,
        "max_det": 300,
        # 数据增强（简化）
        "mosaic": 1.0,
        "close_mosaic": 0,
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
        # 验证配置（使用测试集作为验证集）
        "use_test_as_val": True,
        # 输出配置
        "project": str(PROJECT_DIR),
        "name": NAME,
        "exist_ok": True,
        "save": True,
        "val": True,
        "plots": False,
        # 可复现性
        "deterministic": True,
        "seed": 0,
    }

    print(f"[QuickCheck] 启动快速验证训练（1 epoch，10% 数据，测试集作为验证集）...")
    print(f"[QuickCheck] 数据配置: {DATA_YAML}")
    print(f"[QuickCheck] 主干迁移权重: {PRETRAINED_WEIGHTS}")

    # -------------------------------------------------------------------------
    # 启动训练
    # -------------------------------------------------------------------------
    trainer = IR_OBB_Trainer(overrides=overrides)
    trainer.train()

    print("[QuickCheck] 快速验证训练完成，代码运行正常")


if __name__ == "__main__":
    main()
