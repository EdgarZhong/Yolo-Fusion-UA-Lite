"""
IR 单模态 YOLOv8n OBB 快速验证脚本（COCO 预训练）

使用标准 YOLOv8n OBB 架构（3 通道输入），COCO 预训练权重初始化。
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
from ultralytics.nn.tasks import OBBModel


class IR_OBB_Trainer(OBBTrainer):
    """
    IR 单模态 OBB 训练器：
    - 重写 get_model，构建 3 通道输入的标准 YOLOv8n OBB 模型
    - 加载 COCO 预训练权重 yolov8n-obb.pt 进行初始化
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        构建 3 通道标准 YOLOv8n OBB 模型，并加载预训练权重
        """
        # 使用标准 OBBModel，3 通道输入
        model = OBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose)
        # 加载预训练权重（如果提供）
        if weights:
            model.load(weights)
        return model


def main():
    """
    快速验证训练流程（1 epoch，10% 数据，使用测试集作为验证集）
    """
    # -------------------------------------------------------------------------
    # 配置路径与参数
    # -------------------------------------------------------------------------
    # COCO 预训练权重（YOLOv8n-obb）
    PRETRAINED_WEIGHTS = "yolov8n-obb.pt"

    # 数据集配置（IR 单模态，指向 data_croped/ 目录）
    DATA_YAML = ROOT / "src" / "cfg" / "datasets" / "ir_obb_dronevehicle.yaml"

    # 快速验证输出目录
    PROJECT_DIR = ROOT / "models" / "baseline"
    NAME = "IR-Only-Pretrained-Quick"

    # -------------------------------------------------------------------------
    # 构建训练参数覆盖字典
    # -------------------------------------------------------------------------
    overrides = {
        # 任务与核心路径
        "task": "obb",
        "model": PRETRAINED_WEIGHTS,        # 预训练权重作为模型配置
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
    print(f"[QuickCheck] 预训练权重: {PRETRAINED_WEIGHTS}")

    # -------------------------------------------------------------------------
    # 启动训练
    # -------------------------------------------------------------------------
    trainer = IR_OBB_Trainer(overrides=overrides)
    trainer.train()

    print("[QuickCheck] 快速验证训练完成，代码运行正常")


if __name__ == "__main__":
    main()
