"""
提升 Recall 的最终微调训练脚本（Regularization + Loss Reweighting）

策略依据：`提升recall的最终微调.md`
- 以 Tuned 阶段最佳权重作为热启动，分辨率回归至 640。
- 强正则化：`dropout=0.15`、`label_smoothing=0.1`、`weight_decay=0.001`、`warmup_epochs=0`。
- 损失重加权：`cls=1.5`、`box=8.5`、`fl_gamma=1.5`。
- 真实分布适应增强：关闭 `mosaic/mixup/HSV/degrees/flipud`，保留 `fliplr=0.5/translate=0.1/scale=0.2`。
- 优化器：`SGD`，`lr0=0.001`，`lrf=0.05`。

使用示例：
- `python src/trainning/final_recall_train.py --device 0 --batch 16`
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

# 保障本地 ultralytics 源码可被导入
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBTrainer  # noqa: E402


# ===================== 默认资产与策略参数 =====================
# 源权重（Tuned 阶段最佳）
DEFAULT_WEIGHTS = ROOT / "models" / "posttrain" / "FA-Concat_FPN-PAN_tuned" / "weights" / "best.pt"
# 数据集配置（裁切后 640×512）
DATA_CFG = ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml"

# 训练设置（可通过命令行覆盖 batch/device）
EPOCHS = 30
IMG_SIZE = 640
BATCH_DEFAULT = 16
WORKERS = 2

# 正则化与损失重加权
DROPOUT = 0.15
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 0.001
WARMUP_EPOCHS = 0
CLS_GAIN = 1.5
BOX_GAIN = 8.5
DFL_GAIN = 1.5

# 优化器与学习率策略
OPTIMIZER = "SGD"
LR0 = 0.001
LRF = 0.05
COS_LR = True

# 增强（真实分布适应）
MOSAIC = 0.0
MIXUP = 0.0
HSV_H = 0.0
HSV_S = 0.0
HSV_V = 0.0
DEGREES = 0.0
FLIPLR = 0.5
FLIPUD = 0.0
TRANSLATE = 0.1
SCALE = 0.2

# 输出目录与实验名
PROJECT_DIR = ROOT / "models" / "posttrain"
RUN_NAME = "Final_Recall_640_Regularized_val_on_test"
SAVE_PERIOD = 5


def main():
    """
    启动 Recall 微调训练：
    - 加载 Tuned 最佳权重作为热启动；
    - 应用强正则化与损失重加权；
    - 关闭复杂增强并在 640 分辨率上训练；
    - 每 5 轮保存一次权重。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="运行设备：cpu 或 GPU 索引，如 0/1")
    parser.add_argument("--batch", type=int, default=BATCH_DEFAULT, help="批大小（建议 16，视显存调整）")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="源权重：Tuned 阶段最佳 .pt")
    parser.add_argument("--use-test-as-val", action="store_true", help="训练阶段使用测试集作为验证集")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"未找到源权重文件：{weights_path}")

    overrides = {
        # 任务与核心路径
        "task": "obb",
        "model": str(weights_path),
        "data": str(DATA_CFG),
        # 训练规模与资源
        "epochs": EPOCHS,
        "batch": int(args.batch),
        "workers": WORKERS,
        "device": args.device,
        # 输入与评估配置
        "imgsz": IMG_SIZE,
        "rect": False,
        # 正则化与损失重加权
        "dropout": DROPOUT,
        "label_smoothing": LABEL_SMOOTHING,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "cls": CLS_GAIN,
        "box": BOX_GAIN,
        "dfl": DFL_GAIN,
        # 优化器与学习率
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "lrf": LRF,
        "cos_lr": COS_LR,
        # 增强（真实分布适应）
        "mosaic": MOSAIC,
        "mixup": MIXUP,
        "hsv_h": HSV_H,
        "hsv_s": HSV_S,
        "hsv_v": HSV_V,
        "degrees": DEGREES,
        "fliplr": FLIPLR,
        "flipud": FLIPUD,
        "translate": TRANSLATE,
        "scale": SCALE,
        # 保存策略与位置
        "project": str(PROJECT_DIR),
        "name": RUN_NAME,
        "save": True,
        "save_period": SAVE_PERIOD,
        "val": True,
        "plots": False,
        # 验证阶段统一标准（适配本文策略）：
        "conf": 0.25,
    }

    # 自定义参数：训练阶段强制使用测试集作为验证集
    if bool(args.use_test_as_val):
        overrides["use_test_as_val"] = True

    # 启动信息打印（日志核查）
    print("[Train][OBB] Recall 微调训练启动：")
    print(f"weights={weights_path.as_posix()}")
    print(
        f"epochs={EPOCHS}, imgsz={IMG_SIZE}, batch={int(args.batch)}, workers={WORKERS}, device={args.device}, save_period={SAVE_PERIOD}"
    )
    print(
        f"dropout={DROPOUT}, label_smoothing={LABEL_SMOOTHING}, weight_decay={WEIGHT_DECAY}, warmup_epochs={WARMUP_EPOCHS}, "
        f"cls={CLS_GAIN}, box={BOX_GAIN}, dfl={DFL_GAIN}"
    )
    print(f"optimizer={OPTIMIZER}, lr0={LR0}, lrf={LRF}, cos_lr={COS_LR}")
    print(f"mosaic={MOSAIC}, mixup={MIXUP}, hsv=(h:{HSV_H}, s:{HSV_S}, v:{HSV_V}), degrees={DEGREES}")
    print(f"fliplr={FLIPLR}, flipud={FLIPUD}, translate={TRANSLATE}, scale={SCALE}")
    print(f"project={(PROJECT_DIR / RUN_NAME).as_posix()}")

    trainer = OBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
