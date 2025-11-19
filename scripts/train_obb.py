#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主执行脚本（针对 ultralytics-8.2）：在本地稳定环境中以 Python API 启动 OBB 训练/验证。

设计目标：
- 避免使用 CLI（如 yolo checks），直接用 Python API 提升兼容性与可控性。
- 支持按需生成运行时数据集 YAML（包含 task: obb 与 channels），便于快速验证。
- 针对当前项目的数据目录结构进行自检与友好提示（推荐使用 images/labels 规范）。

使用方式（PowerShell）：
  - 先执行 Conda 初始化与环境激活（详见 README）：
    & "C:\DevLib\miniconda3\Scripts\conda.exe" shell.powershell hook | Out-String | Invoke-Expression
    conda activate .\.conda\ultra82-py312

  - 启动训练示例：
    python scripts/train_obb.py --data-root data --epochs 1 --batch 8 --imgsz 832 --device 0 --channels 6 --name baseline_obb82

注意：
- 若数据集目录命名为 trainimg/valimg/testimg 与 trainlabels_yolo_obb 等非标准形式，建议迁移为 images/labels 规范；
  否则需创建软链接或拷贝少量样本至 images/labels 以进行 smoke test。
"""

import argparse
import sys
from pathlib import Path

# 仅依赖 ultralytics-8.2 本地源码（可编辑安装）
from ultralytics import YOLO


def detect_image_label_layout(data_root: Path):
    """
    自检并返回图像/标签目录布局信息。

    返回：
    - layout: dict，包含 train/val/test 的 images_dir 与 labels_dir（可能为 None）
    - recommend_standard: bool，是否建议使用标准 images/labels 命名
    """
    subsets = ["train", "val", "test"]
    layout = {}
    recommend_standard = False
    for s in subsets:
        base = data_root / s
        std_images = base / "images"
        std_labels = base / "labels"
        alt_images = base / f"{s}img"
        alt_labels = base / f"{s}labels_yolo_obb"
        images_dir = None
        labels_dir = None
        if std_images.exists():
            images_dir = std_images
        elif alt_images.exists():
            images_dir = alt_images
            recommend_standard = True
        if std_labels.exists():
            labels_dir = std_labels
        elif alt_labels.exists():
            labels_dir = alt_labels
            recommend_standard = True
        layout[s] = {"images": images_dir, "labels": labels_dir}
    return layout, recommend_standard


def build_runtime_yaml(data_root: Path, channels: int, names: list[str], save_path: Path):
    """
    生成运行时数据集 YAML 文件（包含 task: obb 与 channels），路径指向当前数据目录。
    - data_root: 数据根目录（包含 train/val/test 子目录）
    - channels: 输入通道数（3 或 6）
    - names: 类别名称列表
    - save_path: YAML 写入目标路径
    """
    layout, recommend = detect_image_label_layout(data_root)
    # 选择 images 目录（若缺失则写入空串，Ultralytics 会报错，提示用户修正）
    train_dir = layout["train"]["images"] or ""
    val_dir = layout["val"]["images"] or ""
    test_dir = layout["test"]["images"] or ""

    # 写入 YAML 文本（names 与 nc）
    names_yaml = "\n".join([f"  {i}: {n}" for i, n in enumerate(names)])
    text = (
        f"path: {data_root.as_posix()}\n"
        f"train: {train_dir}\n"
        f"val: {val_dir}\n"
        f"test: {test_dir}\n"
        f"task: obb\n"
        f"channels: {channels}\n"
        f"names:\n{names_yaml}\n"
        f"nc: {len(names)}\n"
    )
    save_path.write_text(text, encoding="utf-8")
    return recommend


def main():
    """
    主入口：
    - 读取参数，生成运行时数据集 YAML
    - 构建 OBB 模型并执行短轮训练/验证（可选）
    """
    parser = argparse.ArgumentParser(description="YOLOv8‑OBB 训练入口（ultralytics‑8.2）")
    parser.add_argument("--data-root", type=str, default="data", help="数据根目录，含 train/val/test")
    parser.add_argument("--names", type=str, nargs="*", default=["car", "truck", "bus", "van", "freight_car"], help="类别名称列表")
    parser.add_argument("--channels", type=int, default=3, choices=[3, 6], help="输入通道数（3 或 6）")
    parser.add_argument("--model", type=str, default="ultralytics/cfg/models/v8/yolov8-obb.yaml", help="模型配置 YAML 路径")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮次（短轮验证用）")
    parser.add_argument("--batch", type=int, default=8, help="批大小")
    parser.add_argument("--imgsz", type=int, default=832, help="输入分辨率（正方形或最大边）")
    parser.add_argument("--device", type=int, default=0, help="GPU 编号（或 -1 使用 CPU）")
    parser.add_argument("--workers", type=int, default=2, help="dataloader 进程数")
    parser.add_argument("--name", type=str, default="baseline_obb82", help="保存目录名称（位于 models/ 下）")
    parser.add_argument("--val-only", action="store_true", help="仅执行验证，不训练")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    runtime_yaml = Path("models") / f"runtime_obb_{args.channels}ch.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)

    # 生成 YAML 并提示目录规范建议
    recommend = build_runtime_yaml(data_root, args.channels, args.names, runtime_yaml)
    if recommend:
        print("[提示] 建议将数据目录迁移为标准命名：images/labels（当前检测到非标准命名，将可能导致自动标签映射失败）")

    # 构建并执行
    model = YOLO(args.model)
    if args.val_only:
        res = model.val(data=str(runtime_yaml), imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers)
        print("验证完成：", res)
    else:
        model.train(
            data=str(runtime_yaml),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            name=args.name,
            project="models",
            task="obb",
        )
        print("训练完成，日志与权重位于 models/ 目录")


if __name__ == "__main__":
    sys.exit(main())