"""
IR 单模态 YOLOv8n OBB 快速验证脚本（COCO 预训练）

用于验证：
- 数据集路径配置正确
- COCO 预训练权重加载正常
- 单模态 IR 训练流程可正常运行

验证通过后，使用 ir_only_pretrained_baseline.py 进行全量训练。
"""

import sys
from pathlib import Path

# ================== 仓库根路径导入 ==================
ROOT = Path(__file__).resolve().parents[2]


def main():
    """
    快速验证训练流程

    使用 10% 数据子集训练 2 个 epoch，快速验证代码正确性。
    """
    # 导入 Ultralytics（使用 editable install 的本地版本）
    from ultralytics import YOLO

    # -------------------------------------------------------------------------
    # 1. 配置路径与参数
    # -------------------------------------------------------------------------
    # COCO 预训练权重（YOLOv8n-obb，ultralytics 会自动下载如果不存在）
    # 注意：必须使用 -obb 版本，因为 OBB 任务使用旋转框格式（xywhr），与标准检测（xyxy）不同
    PRETRAINED_WEIGHTS = "yolov8n-obb.pt"

    # 数据集配置（IR 单模态，指向 data_croped/ 目录）
    DATA_YAML = ROOT / "src" / "cfg" / "datasets" / "ir_obb_dronevehicle.yaml"

    # 快速验证输出目录（与正式训练分开）
    PROJECT_DIR = ROOT / "models" / "baseline"
    NAME = "IR-Only-Pretrained-Quick"

    # -------------------------------------------------------------------------
    # 2. 加载 COCO 预训练模型
    # -------------------------------------------------------------------------
    print("[QuickCheck] 正在加载 COCO 预训练权重 yolov8n-obb.pt...")
    model = YOLO(PRETRAINED_WEIGHTS)
    print("[QuickCheck] 模型加载成功")

    # -------------------------------------------------------------------------
    # 3. 启动快速验证训练
    # -------------------------------------------------------------------------
    print("[QuickCheck] 启动快速验证训练（2 epoch，10% 数据）...")
    model.train(
        # 任务与数据配置
        task="obb",
        data=DATA_YAML,

        # 快速验证规模（小规模）
        epochs=2,
        batch=16,
        imgsz=640,
        device=0,
        workers=0,                           # Windows 下使用 0 避免多进程问题
        fraction=0.1,                        # 仅使用 10% 数据子集

        # 训练策略（快速验证不冻结）
        patience=0,
        freeze=0,                            # 快速验证不冻结 backbone
        rect=False,

        # 验证参数
        conf=0.25,
        iou=0.75,
        max_det=300,

        # 数据增强（简化）
        mosaic=1.0,
        close_mosaic=0,                      # 快速验证不关 mosaic
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        # 输出配置
        project=PROJECT_DIR,
        name=NAME,
        exist_ok=True,
        save=True,
        val=True,
        plots=False,                         # 快速验证不生成图表

        # 可复现性
        deterministic=True,
        seed=0,
    )

    print("[QuickCheck] 快速验证训练完成，代码运行正常")


if __name__ == "__main__":
    main()
