"""
快速验证训练脚本（阶段一 / 基线）

功能：
- 使用我们在 `ultralytics-8.2/` 官方源码上扩展的双主干 OBB 模型与双路数据集加载逻辑，
  对 DroneVehicle 数据集进行一次最小开销的验证训练（1 个 epoch，0.05 训练数据比例）。

注意：
- 为避免外部环境未执行 `pip install -e ultralytics-8.2`，脚本在运行时将该目录加入 `sys.path` 以确保可导入。
- 基线权重与日志将输出到 `models/baseline/dualbackbone-easy-obb-baseline/`。
"""

import os
import sys
from pathlib import Path

# 保障本地源码可被导入
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBTrainer  # noqa: E402
import yaml  # noqa: E402


def main():
    """
    启动一次快速验证训练：
    - 模型：`src/cfg/model/dualbackbone_easy_obb.yaml`（双主干 + 基础融合）
    - 数据：`src/cfg/datasets/dual_obb_dronevehicle.yaml`（双路目录与标签映射约定）
    - 训练：1 轮次；训练数据比例 0.05；禁用数据增强；矩形评估；分辨率设为 832（stride=32 对齐）
    """

    model_cfg = str(ROOT / "src" / "cfg" / "model" / "dualbackbone_easy_obb.yaml")
    data_cfg = str(ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml")

    overrides = {
        # 任务与核心路径
        "task": "obb",
        "model": model_cfg,
        "data": data_cfg,
        # 训练规模与资源
        "epochs": 1,
        "batch": 8,
        "workers": 2,
        "device": 0,
        # 数据比例（快速验证）
        "fraction": 0.05,
        # 输入与评估配置
        "imgsz": 832,  # 训练/验证统一为 832 尺寸
        "rect": False,  # 训练阶段关闭矩形分桶；验证/测试阶段由内部逻辑自动启用矩形
        # 关闭增强，保持原生分布
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 1.0,
        "shear": 0.0,
        # 保存位置（按 README 要求）
        "project": str(ROOT / "models" / "baseline"),
        "name": "dualbackbone-easy-obb-baseline",
        "save": True,
        "val": True,
        "patience": 0,
        "plots": False,
    }

    print("[Train][OBB] 启动快速验证训练（真实数据）：epochs=1, fraction=0.05, imgsz=832, rect=False(train)/rect=True(val)")
    print(f"[Train][OBB] 模型配置: {model_cfg}")
    print(f"[Train][OBB] 数据配置: {data_cfg}")

    trainer = OBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()