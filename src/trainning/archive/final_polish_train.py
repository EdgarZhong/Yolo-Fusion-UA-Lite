"""
最终微调训练脚本（High-Res Fine-tuning，基于双主干 OBB 模型）

目标与约束：
- 在上一阶段调优得到的最佳权重基础上，进行高分辨率（imgsz=800）短周期微调，进一步提升小目标召回与总体精度。
- 严格关闭所有“非真实样本”增强（mosaic/mixup/copy_paste/HSV/旋转等），仅保留极小几何扰动（translate/scale）与水平翻转，以避免破坏已收敛的特征结构。
- 使用 SGD 优化器，并将初始学习率设为 0.0025，配合余弦退火与较快衰减（lrf=0.05），实现平滑接续训练。
- 每 5 轮保存一次检查点（save_period=5），以便出现抖动时回退与对比。

使用示例：
- GPU 训练：`python src/trainning/final_polish_train.py --device 0`
- 指定 batch：`python src/trainning/final_polish_train.py --device 0 --batch 12`
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


# ===================== 默认资产路径与超参 =====================
# 上一阶段的最佳权重（Tuned 阶段输出）
DEFAULT_WEIGHTS = ROOT / "models" / "posttrain" / "FA-Concat_FPN-PAN_tuned" / "weights" / "best.pt"
# 数据集配置（保持与项目约定一致）
DATA_CFG = ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml"

# 关键训练设置（可通过命令行覆盖 batch/device）
EPOCHS = 40
IMG_SIZE = 800
BATCH_DEFAULT = 12  # 高分辨率显存占用较高，必要时降低为 8/4
WORKERS = 2
OPTIMIZER = "SGD"
LR0 = 0.0025
LRF = 0.05
WARMUP_EPOCHS = 0
COS_LR = True

# 增强策略（高分辨率纯净微调）
MOSAIC = 0.0
MIXUP = 0.0
COPY_PASTE = 0.0
HSV_H = 0.0
HSV_S = 0.0
HSV_V = 0.0
DEGREES = 0.0
FLIPLR = 0.5
FLIPUD = 0.0
TRANSLATE = 0.05
SCALE = 0.1

# 输出目录与实验名
PROJECT_DIR = ROOT / "models" / "posttrain"
RUN_NAME = "Final_Polish_800_FlipScale"
SAVE_PERIOD = 5


def main():
    """
    启动最终微调训练：
    - 加载已训练的最佳权重；
    - 指定高分辨率与纯净增强策略；
    - 使用 SGD + 余弦退火进行短周期微调；
    - 每 5 轮保存一次权重，便于快速回退与比较。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="运行设备：cpu 或 GPU 索引，如 0/1")
    parser.add_argument("--batch", type=int, default=BATCH_DEFAULT, help="批大小（高分辨率建议 12/8/4）")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="源权重：上一阶段最佳权重 .pt")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"未找到源权重文件：{weights_path}")

    overrides = {
        # 任务与核心路径
        "task": "obb",
        "model": str(weights_path),  # 加载 .pt 以继承结构与参数
        "data": str(DATA_CFG),
        # 训练规模与资源
        "epochs": EPOCHS,
        "batch": int(args.batch),
        "workers": WORKERS,
        "device": args.device,
        # 输入与评估配置
        "imgsz": IMG_SIZE,
        "rect": False,  # 训练阶段保持 shuffle=True 的非矩形采样
        # 优化器与学习率策略
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "lrf": LRF,
        "warmup_epochs": WARMUP_EPOCHS,
        "cos_lr": COS_LR,
        # 增强配置（纯净微调）
        "mosaic": MOSAIC,
        "mixup": MIXUP,
        "copy_paste": COPY_PASTE,
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
    }

    # 启动信息打印（用于日志复核）
    print("[Train][OBB] 最终微调训练启动：")
    print(f"weights={weights_path.as_posix()}")
    print(
        f"epochs={EPOCHS}, imgsz={IMG_SIZE}, batch={int(args.batch)}, workers={WORKERS}, device={args.device}, save_period={SAVE_PERIOD}"
    )
    print(f"optimizer={OPTIMIZER}, lr0={LR0}, lrf={LRF}, warmup_epochs={WARMUP_EPOCHS}, cos_lr={COS_LR}")
    print(f"mosaic={MOSAIC}, mixup={MIXUP}, copy_paste={COPY_PASTE}, hsv=(h:{HSV_H}, s:{HSV_S}, v:{HSV_V})")
    print(f"degrees={DEGREES}, fliplr={FLIPLR}, flipud={FLIPUD}, translate={TRANSLATE}, scale={SCALE}")
    print(f"project={(PROJECT_DIR / RUN_NAME).as_posix()}")

    trainer = OBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()

