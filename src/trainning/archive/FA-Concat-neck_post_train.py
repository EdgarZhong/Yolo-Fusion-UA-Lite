import sys
from pathlib import Path

# ================== 仓库根路径与 Ultralytics 源码导入 ==================
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

import argparse
import os
import torch

from ultralytics.models.yolo.obb import OBBTrainer


"""
宏定义默认参数（与正式训练脚本风格一致）
"""
# 训练超参数（可通过 CLI 覆盖）
EPOCHS = 200
BATCH = 12
WORKERS = 2
DEVICE = 0
FRACTION = 1.0
IMG_SIZE = 640
OPTIMIZER = "SGD"
LR0 = 0.01
FREEZE = 10
PATIENCE = 25
# 生命周期控制（默认与实施手册一致：最后 25 轮关闭）
CLOSE_MOSAIC = 25
CLOSE_DROPOUT = 25
# 训练期按 epoch 控制 Backbone 冻结/解冻
UNFREEZE_EPOCH = 10
# 轻量几何增强
TRANSLATE = 0.1
SCALE = 0.5
FLIPLR = 0.5
FLIPUD = 0.0
MOSAIC = 1.0
# 模态随机 Dropout 概率（默认各 0.10）
DROP_PROB_RGB = 0.10
DROP_PROB_IR = 0.10


def build_overrides(args: argparse.Namespace) -> dict:
    """
    构造 Ultralytics 训练参数字典（overrides）。

    关键设计：
    - 使用 FA-Concat + FPN/PAN 颈部配置（或热启动权重文件）；
    - 将模态随机 Dropout 概率作为训练参数（`drop_prob_rgb`/`drop_prob_ir`）注入；
    - 在最后 `close_mosaic` 轮自动关闭 Mosaic，同时关闭模态 Dropout（生命周期控制）。
    """

    # 必须加载迁移生成的热启动权重，未找到则报错
    migratory = ROOT / "models" / "migratory" / "dualbackbone_FA_Concat_pretrained.pt"
    if not migratory.exists():
        raise FileNotFoundError(
            f"未找到热启动权重文件: {migratory}，请先运行 'python src/tools/transfer_dual_weights.py' 生成。"
        )
    model_cfg = str(migratory)

    data_cfg = str(ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml")

    overrides = {
        # 任务与核心路径
        "task": "obb",
        "model": model_cfg,
        "data": data_cfg,
        # 训练规模与资源
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        # 数据比例（正式训练建议设为 1.0；快速验证可设置 < 1）
        "fraction": args.fraction,
        # 输入与评估配置
        "imgsz": args.imgsz,
        "rect": False,
        # 优化器与学习率（与实施手册一致）
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        # 关闭 Ultralytics 原生按层冻结，改为脚本按 epoch 控制
        "freeze": 0,
        # 增强配置：关闭 HSV/旋转，维持主分布稳定；Mosaic 在尾期自动关闭
        "mosaic": args.mosaic,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": args.translate,
        "scale": args.scale,
        "shear": 0.0,
        # 输出目录与命名（实施手册约定路径）
        "project": str(ROOT / "models" / "posttrain"),
        "name": "FA-Concat_FPN-PAN_tuned",
        "save": True,
        "val": True,
        "patience": args.patience,
        "plots": False,
        # 生命周期控制：最后 close_mosaic 轮同时关闭 Mosaic 与 模态 Dropout
        "close_mosaic": args.close_mosaic,
    }

    return overrides


def inject_modality_dropout(trainer: OBBTrainer, drop_prob_rgb: float, drop_prob_ir: float, close_dropout: int):
    """
    在 Trainer 的 batch 预处理阶段注入“模态随机 Dropout”逻辑。

    行为与约束：
    - 仅对 6 通道输入生效（前三通道 RGB，后三通道 IR）；
    - 每个样本独立以 `drop_prob_rgb`/`drop_prob_ir` 概率置零相应模态；
    - 在最后 `close_mosaic` 轮自动关闭（与 Mosaic 生命周期保持一致）。
    """

    orig = trainer.preprocess_batch

    def wrapped(batch: dict) -> dict:
        b = orig(batch)
        # 生命周期：在最后 close_dropout 轮关闭 Dropout
        if trainer.epoch < (trainer.epochs - close_dropout):
            imgs = b.get("img")
            if imgs is not None and imgs.ndim == 4 and imgs.size(1) == 6:
                B = imgs.shape[0]
                # 概率参数（由 CLI 传入，不写入 Ultralytics 配置）
                p_r = float(drop_prob_rgb or 0.0)
                p_i = float(drop_prob_ir or 0.0)
                if p_r > 0.0:
                    mask_r = torch.rand(B, device=imgs.device) < p_r
                    if mask_r.any():
                        imgs[mask_r, 0:3, :, :] = 0.0
                if p_i > 0.0:
                    mask_i = torch.rand(B, device=imgs.device) < p_i
                    if mask_i.any():
                        imgs[mask_i, 3:6, :, :] = 0.0
        return b

    trainer.preprocess_batch = wrapped


def inject_backbone_freeze_schedule(trainer: OBBTrainer, unfreeze_epoch: int):
    """
    按 epoch 控制双主干（RGB: 3..12，IR: 13..22）的冻结/解冻：
    - 训练开始：冻结两路主干参数，避免颈部随机初始化导致梯度污染；
    - 到达 `unfreeze_epoch`：自动解冻两路主干，参与后续联合训练。
    """

    def _core_model():
        m = trainer.model
        return m.module if hasattr(m, "module") else m

    def _set_requires_grad(indices, requires_grad: bool):
        model = _core_model()
        for i in indices:
            for p in model.model[i].parameters():
                p.requires_grad = requires_grad

    backbone_indices = list(range(3, 13)) + list(range(13, 23))

    def on_train_start(_):
        _set_requires_grad(backbone_indices, False)

    def on_train_epoch_start(_):
        if trainer.epoch == unfreeze_epoch:
            _set_requires_grad(backbone_indices, True)

    trainer.add_callback("on_train_start", on_train_start)
    trainer.add_callback("on_train_epoch_start", on_train_epoch_start)


def parse_args() -> argparse.Namespace:
    """命令行参数解析：覆盖训练超参与模态 Dropout 概率。"""
    p = argparse.ArgumentParser()
    # 核心训练参数（与实施手册一致）
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--device", type=int, default=DEVICE)
    p.add_argument("--fraction", type=float, default=FRACTION)
    p.add_argument("--imgsz", type=int, default=IMG_SIZE)
    p.add_argument("--optimizer", type=str, default=OPTIMIZER)
    p.add_argument("--lr0", type=float, default=LR0)
    p.add_argument("--freeze", type=int, default=FREEZE)
    p.add_argument("--unfreeze_epoch", type=int, default=UNFREEZE_EPOCH, help="在该 epoch 自动解冻 Backbone")
    p.add_argument("--close_mosaic", type=int, default=CLOSE_MOSAIC)
    p.add_argument("--close_dropout", type=int, default=CLOSE_DROPOUT, help="最后关闭模态 Dropout 的 epoch 数")
    p.add_argument("--patience", type=int, default=PATIENCE)
    # 轻量几何增强（保持与手册的保守设定）
    p.add_argument("--translate", type=float, default=TRANSLATE)
    p.add_argument("--scale", type=float, default=SCALE)
    p.add_argument("--fliplr", type=float, default=FLIPLR)
    p.add_argument("--flipud", type=float, default=FLIPUD)
    p.add_argument("--mosaic", type=float, default=MOSAIC)
    # 模态随机 Dropout 概率（可调）
    p.add_argument("--drop-prob-rgb", dest="drop_prob_rgb", type=float, default=DROP_PROB_RGB)
    p.add_argument("--drop-prob-ir", dest="drop_prob_ir", type=float, default=DROP_PROB_IR)
    return p.parse_args()


def main():
    """
    FA‑Concat + Neck 深度微调训练入口（支持可调模态随机 Dropout）。

    默认配置遵循实施手册：
    - epochs=200、imgsz=640、freeze=10、close_mosaic=25；
    - HSV/旋转增强全部关闭；
    - 模态 Dropout 在前 175 个 epoch 生效，在最后 25 个 epoch 自动关闭。
    """

    args = parse_args()
    overrides = build_overrides(args)

    print("[PostTrain][FA‑Concat+Neck] 启动深度微调训练…")
    print(
        f"epochs={args.epochs}, batch={args.batch}, workers={args.workers}, device={args.device}, fraction={args.fraction}, imgsz={args.imgsz}"
    )
    print(
        f"optimizer={args.optimizer}, lr0={args.lr0}, freeze_schedule=[start_freeze_backbone, unfreeze_epoch={args.unfreeze_epoch}], close_mosaic={args.close_mosaic}, close_dropout={args.close_dropout}, drop_prob_rgb={args.drop_prob_rgb}, drop_prob_ir={args.drop_prob_ir}"
    )
    print(f"model={overrides['model']}")
    print(f"data={overrides['data']}")
    print(f"save_dir={(ROOT / 'models' / 'posttrain' / 'FA-Concat_FPN-PAN_tuned').as_posix()}")

    trainer = OBBTrainer(overrides=overrides)
    inject_modality_dropout(trainer, args.drop_prob_rgb, args.drop_prob_ir, args.close_dropout)
    inject_backbone_freeze_schedule(trainer, args.unfreeze_epoch)
    trainer.train()


if __name__ == "__main__":
    main()
