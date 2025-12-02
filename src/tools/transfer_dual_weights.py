import sys
from pathlib import Path

# ============ 仓库根路径与 Ultralytics 源码导入 ============
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

import argparse
import torch

from ultralytics.nn.tasks import DualBackboneOBBModel, attempt_load_one_weight


def deep_copy_module_params(src: torch.nn.Module, dst: torch.nn.Module) -> int:
    """
    逐张量深拷贝参数（包含权重与偏置），仅在形状完全一致时覆盖。

    返回成功拷贝的张量数量，便于统计与日志打印。
    """
    sc = src.state_dict()
    dc = dst.state_dict()
    copied = 0
    for k, v in sc.items():
        if k in dc and dc[k].shape == v.shape:
            dc[k].copy_(v)
            copied += 1
    dst.load_state_dict(dc, strict=False)
    return copied


def main():
    """
    将 `yolov8n.pt` 的前 10 层 Backbone 权重分别迁移到双主干 FA‑Concat + Neck 模型的 RGB/IR 两路主干中。

    映射规则（基于本仓库的 YAML 索引约定）：
    - 源：`yolov8n` backbone 层索引 0..9（Conv/C2f/Conv/C2f/Conv/C2f/Conv/C2f/SPPF）
    - 目标：RGB 分支 3..12，IR 分支 13..22（每路 10 层，与源一一对应）
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=str(ROOT / "yolov8n.pt"), help="源权重文件路径 (yolov8n.pt)")
    parser.add_argument(
        "--model-yaml",
        type=str,
        default=str(ROOT / "src" / "cfg" / "model" / "FA_Concat_FPN-PAN_neck.yaml"),
        help="目标双主干模型配置 YAML",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=str(ROOT / "models" / "migratory" / "dualbackbone_FA_Concat_pretrained.pt"),
        help="输出的热启动权重文件路径",
    )
    args = parser.parse_args()

    # 1) 加载源权重（yolov8n）
    print(f"[Transfer] 加载源权重: {args.src}")
    src_model, src_ckpt = attempt_load_one_weight(args.src, device=None, inplace=True, fuse=False)

    # 2) 构建目标模型（FA‑Concat + FPN/PAN Neck），保证 6 通道与 5 类一致
    print(f"[Transfer] 构建目标模型: {args.model_yaml}")
    tgt_model = DualBackboneOBBModel(cfg=args.model_yaml, ch=6, nc=5, verbose=True)

    # 3) 定义索引映射
    src_idx = list(range(0, 10))  # YOLOv8n backbone 层 0..9
    rgb_idx = list(range(3, 13))  # 目标 RGB 主干层 3..12
    ir_idx = list(range(13, 23))  # 目标 IR 主干层 13..22

    # 4) 逐层拷贝（RGB）
    copied_rgb = 0
    for si, ti in zip(src_idx, rgb_idx):
        copied_rgb += deep_copy_module_params(src_model.model[si], tgt_model.model[ti])
    print(f"[Transfer] RGB 分支拷贝完成，复制张量数: {copied_rgb}")

    # 5) 逐层拷贝（IR）
    copied_ir = 0
    for si, ti in zip(src_idx, ir_idx):
        copied_ir += deep_copy_module_params(src_model.model[si], tgt_model.model[ti])
    print(f"[Transfer] IR 分支拷贝完成，复制张量数: {copied_ir}")

    # 6) 写出热启动权重文件
    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_out = {
        "model": tgt_model,
        "train_args": {"task": "obb"},
        "epoch": 0,
        "best_fitness": 0.0,
    }
    torch.save(ckpt_out, dst_path)
    print(f"[Transfer] 已生成迁移权重: {dst_path}")


if __name__ == "__main__":
    main()

