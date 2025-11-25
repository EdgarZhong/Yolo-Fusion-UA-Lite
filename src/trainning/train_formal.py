"""
正式训练脚本（基于双主干 OBB 模型，双模态 RGB+IR，6 通道输入）

设计目标：
- 顶部统一“宏定义”管理所有关键超参与训练集使用比例，便于快速修改与复现实验。
- 数据输入与策略保持与项目约定一致：
  - 输入通道顺序固定为 6 通道，前 3 为 RGB，后 3 为 IR（由数据集与 `ModalitySelector` 保证）。
  - 训练：`imgsz=832`、`rect=False`、开启随机打乱；禁用全部增强（mosaic/mixup/copy_paste/erasing/flip/HSV）。
  - 验证/测试：`imgsz=832`、`rect=True`，其余保持默认，不做额外操作。

运行方式：
- 在 PowerShell 中按 README 激活 Conda 环境后执行：
  - `python src/trainning/train_formal.py`
"""

from pathlib import Path
import sys

# =============== 宏定义区域（可根据需要修改） ===============
# 模型与数据配置路径（相对仓库根目录）
MODEL_CFG = "src/cfg/model/dualbackbone_fusionattention_obb.yaml"
DATA_CFG = "src/cfg/datasets/dual_obb_dronevehicle.yaml"

# 训练超参数
EPOCHS = 150          # 建议：100 ~ 300。对于基线实验，100轮足以观察收敛趋势和性能。
BATCH = 12             # 建议：根据显存最大化。如果显存允许（如24G），尝试16或32以稳定梯度。8是安全起步值。
WORKERS = 2           # 建议：根据CPU核数。通常设为4或8能保证数据加载不成为瓶颈。
DEVICE = 0            # GPU0，保持不变
FRACTION = 1.0        # 正式训练必须为 1.0 (使用全量数据)
IMG_SIZE = 832        # 保持 832。对于 840x712 的原图，这是最佳的 32 倍数填充尺寸。
RECT_TRAIN = False    # 保持 False。确保 Shuffle 开启，对训练至关重要。
PATIENCE = 15         # 早停耐心值（无提升的最大容忍 epoch 数），正式训练可调整

# 优化器与学习率设置（显式配置与显示）
# - OPTIMIZER 支持 "auto"（根据任务自动选择），或显式指定如 "SGD"、"AdamW" 等
# - LR0 为初始学习率（与 Ultralytics 的余弦/线性调度配合使用），此处保持官方默认 0.01
OPTIMIZER = "SGD"
LR0 = 0.01

# 数据增强开关（全部关闭，保持原生分布）
MOSAIC = 0.0
MIXUP = 0.0
COPY_PASTE = 0.0
ERASING = 0.0
FLIPLR = 0.0
FLIPUD = 0.0
HSV_H = 0.0
HSV_S = 0.0
HSV_V = 0.0
DEGREES = 0.0
TRANSLATE = 0.0
SCALE = 1.0
SHEAR = 0.0

# 输出目录与实验名
PROJECT_DIR = "models/formal/fusion-attention"
RUN_NAME = "dualbackbone-fusionattention-obb"

# ==========================================================

# 保障本地源码可被导入（使用仓库内 ultralytics-8.2 源码）
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBTrainer  # noqa: E402


def main():
    """
    启动正式训练：
    - 使用宏定义的统一超参与数据比例；
    - 保持数据输入和禁用增强策略与项目约定一致。
    """

    model_cfg = str(ROOT / MODEL_CFG)
    data_cfg = str(ROOT / DATA_CFG)

    overrides = {
        # 任务与核心路径
        "task": "obb",
        "model": model_cfg,
        "data": data_cfg,
        # 训练规模与资源
        "epochs": EPOCHS,
        "batch": BATCH,
        "workers": WORKERS,
        "device": DEVICE,
        # 数据比例（正式训练建议设为 1.0）
        "fraction": FRACTION,
        # 输入与评估配置
        "imgsz": IMG_SIZE,
        "rect": RECT_TRAIN,
        # 优化器与学习率（显式设置，便于复现与展示）
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        # 关闭增强，保持原生分布
        "mosaic": MOSAIC,
        "mixup": MIXUP,
        "copy_paste": COPY_PASTE,
        "erasing": ERASING,
        "fliplr": FLIPLR,
        "flipud": FLIPUD,
        "hsv_h": HSV_H,
        "hsv_s": HSV_S,
        "hsv_v": HSV_V,
        "degrees": DEGREES,
        "translate": TRANSLATE,
        "scale": SCALE,
        "shear": SHEAR,
        # 保存位置
        "project": str(ROOT / PROJECT_DIR),
        "name": RUN_NAME,
        "save": True,
        "val": True,
        "patience": PATIENCE,
        "plots": False,
        # 稳定性增强：禁用 AMP，启用确定性算法，避免潜在的半精度/核选择问题
        # "amp": False,
        # "deterministic": True,
    }

    print("[Train][OBB] 正式训练启动，统一宏定义参数如下：")
    print(
        f"epochs={EPOCHS}, batch={BATCH}, workers={WORKERS}, device={DEVICE}, fraction={FRACTION}, imgsz={IMG_SIZE}, rect(train)={RECT_TRAIN}"
    )
    print(f"model={model_cfg}")
    print(f"data={data_cfg}")
    # 显式展示优化器与学习率设置，便于在日志中快速确认配置
    print(f"optimizer={OPTIMIZER}, lr0={LR0}")
    print(f"save_dir={(ROOT / PROJECT_DIR / RUN_NAME).as_posix()}")

    trainer = OBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
