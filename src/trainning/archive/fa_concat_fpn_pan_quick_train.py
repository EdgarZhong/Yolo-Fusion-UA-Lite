"""
快速验证训练脚本（阶段三 / FA‑Concat + FPN/PAN Neck）

功能：
- 使用我们在 `ultralytics-8.2/` 官方源码上扩展的双主干 OBB 模型，与新补全的 FPN+PAN 颈部架构，
  对 DroneVehicle 数据集进行一次最小开销的验证训练（1 个 epoch，0.05 训练数据比例），以验证模型配置正确性。

注意：
- 为避免外部环境未执行 `pip install -e ultralytics-8.2`，脚本在运行时将该目录加入 `sys.path` 以确保可导入。
- 权重与日志输出到 `models/fusion-attention/dualbackbone-FA-Concat-FPN-PAN-obb-quick/`，与阶段二的 FA‑Concat 快速验证相区分。
"""

import sys
from pathlib import Path

# 保障本地源码可被导入（优先使用仓库内 ultralytics-8.2 源码）
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBTrainer  # noqa: E402


def main():
    """
    启动一次快速验证训练：
    - 模型：`src/cfg/model/FA-Concat-FPN-PAN-neck.yaml`（FA‑Concat 融合 + FPN/PAN 完整颈部）
    - 数据：`src/cfg/datasets/dual_obb_dronevehicle.yaml`（双路目录与标签映射约定）
    - 训练：1 轮次；训练数据比例 0.05；禁用数据增强；训练 `rect=False`；分辨率设为 832（stride=32 对齐）。
    - 目的：验证模型 YAML 可被正确解析与构建，训练流程可正常跑通。
    """

    # 路径配置（相对仓库根目录）
    model_cfg = str(ROOT / "src" / "cfg" / "model" / "FA-Concat-FPN-PAN-neck.yaml")
    data_cfg = str(ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml")

    # 训练覆盖参数（快速验证最小化设置）
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
        # 保存位置与命名：与阶段二保持同一目录层级，名称体现 Neck 补全
        "project": str(ROOT / "models" / "fusion-attention"),
        "name": "dualbackbone-FA-Concat-FPN-PAN-obb-quick",
        "save": True,
        "val": True,
        "patience": 0,
        "plots": False,
    }

    print("[Train][OBB] 启动 FA‑Concat + FPN/PAN 快速验证训练：epochs=1, fraction=0.05, imgsz=832, rect=False(train)/rect=True(val)")
    print(f"[Train][OBB] 模型配置: {model_cfg}")
    print(f"[Train][OBB] 数据配置: {data_cfg}")

    trainer = OBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()

