"""
通用测试脚本（不做可视化保存）：在测试集上评估 OBB 模型并输出性能数据

功能概述（符合项目与 ultralytics-8.2 的改动）：
- 仅执行基本推理/评估流程，输出总体与每类指标到 JSON/CSV；不绘制/保存任何推理图片。
- 默认使用 GPU（`--device 0`），可指定权重路径、模型名称（用于结果命名）与结果输出路径。
- 支持按测试集比例进行评估（`--test-ratio`，例如 0.2 表示使用 20% 测试样本）。

使用方式（在项目根目录激活环境后）：
- `python src/testing/test_baseline.py --device 0 --test-ratio 1.0 --model-name baseline --result-dir result`
- 可选参数：
  - `--device cpu|0|1` 指定运行设备；默认 `0`（GPU）
  - `--weights <path>` 指定权重文件或包含 `weights/` 的目录（未提供则自动从默认 formal 目录中查找）
  - `--model-name <str>` 结果命名（用于 `*.json/*.csv` 文件名）
  - `--result-dir <path>` 结果输出目录（默认 `result`）
  - `--test-ratio <float>` 测试集比例（0-1]，默认 1.0；按比例子集评估以加速

说明：
- 所有路径均以仓库根目录为基准；不执行任何可视化图片保存；评估时采用矩形分桶与固定分辨率。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import argparse

import torch

# ====== 保证导入本仓库随附的 ultralytics-8.2 源码 ======
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBValidator  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.data import build_yolo_dataset  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


# ===================== 路径与常量约定 =====================
DATA_CFG = ROOT / "src/cfg/datasets/dual_obb_dronevehicle.yaml"  # 裁切后 640×512 的数据集 YAML（绝对路径）
WEIGHTS_DIR = ROOT / "models/formal/posttrain/Final_Recall_640_Regularized"  # Recall 微调输出目录
DEFAULT_RESULT_DIR = ROOT / "result"  # 默认评估输出根目录
DEFAULT_RUN_NAME = "Final_Recall_640_Regularized"  # 默认评估结果文件名（不含扩展名）

IMG_SIZE = 640  # 裁切后数据集统一分辨率宽度，配合 rect=True 保持 640x512 形状
BATCH = 16  # 可根据显存/CPU性能调整
CONF_THRES = 0.25  # 统一测试标准：置信度阈值 0.25
IOU_THRES = 0.75
MAX_DET = 500  # 统一测试标准：每图最大检测数 500


def _find_weights(base: Path) -> Path:
    """
    在给定的训练输出目录下寻找权重文件。

    优先返回 `weights/best.pt`，若不存在则返回 `weights/last.pt`；再无则在 `weights/` 下寻找任意 `.pt`。
    """
    wdir = base / "weights"
    candidates = [wdir / "best.pt", wdir / "last.pt"]
    for p in candidates:
        if p.exists():
            return p
    other = next(iter(wdir.glob("*.pt")), None)
    if other is None:
        raise FileNotFoundError(f"未找到权重文件：{wdir.as_posix()} 下不存在 best.pt/last.pt")
    return other


def run_eval(
    weights: Path,
    device: str = "0",
    data_cfg: Path = DATA_CFG,
    result_dir: Path | None = None,
    run_name: str | None = None,
    test_ratio: float = 1.0,
    test_aug: bool = True,
    iou: float = IOU_THRES,
    max_det: int = MAX_DET,
) -> Path:
    """
    运行 OBB 验证流程（split=test），保存评估指标到 指定目录（默认 result/）
    """
    # 结果目录与命名处理：若未指定则使用默认；输出统一到 result/<模型目录>/<文件>
    result_root = Path(result_dir) if result_dir else DEFAULT_RESULT_DIR
    result_root.mkdir(parents=True, exist_ok=True)
    name = run_name or DEFAULT_RUN_NAME
    out_dir = result_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 组装验证参数（保持与项目约定一致）
    # 根据设备类型设置 workers（CPU 场景下置 0 更稳定）
    workers = 0 if str(device).lower() == "cpu" else 2

    # 组装验证参数（保持与项目约定一致）
    args = dict(
        task="obb",
        model=str(weights),
        data=str(data_cfg),
        split="test",
        imgsz=IMG_SIZE,
        rect=True, #测试时启用长方形输入
        batch=BATCH,
        workers=workers,
        device=device,
        plots=False,
        save_json=True,
        conf=CONF_THRES,
        iou=float(iou),
        augment=bool(test_aug),
        max_det=int(max_det),
        project=str(result_root),
        name=name,
    )

    # 若设置测试集比例（0-1]，则自建 DataLoader 并按比例裁剪样本数量，以加速评估
    if isinstance(test_ratio, float) and 0.0 < test_ratio < 1.0:
        data = check_det_dataset(str(data_cfg))
        cfg = get_cfg(overrides=dict(task="obb", imgsz=IMG_SIZE, rect=True))
        base_ds = build_yolo_dataset(cfg, data.get("test"), BATCH, data, mode="val")

        # 计算子集大小，至少为 1
        count = max(1, int(len(base_ds) * test_ratio))
        from torch.utils.data import DataLoader, Subset

        indices = list(range(count))
        subset = Subset(base_ds, indices)
        loader = DataLoader(
            dataset=subset,
            batch_size=min(BATCH, len(indices)),
            shuffle=False,
            num_workers=workers,
            collate_fn=getattr(base_ds, "collate_fn", None),
            pin_memory=False,
        )
        validator = OBBValidator(dataloader=loader, args=args)
        print(f"[Eval][OBB] 已创建验证器（子集）并透传 max_det={getattr(validator.args, 'max_det', None)}")
    else:
        validator = OBBValidator(args=args)
        print(f"[Eval][OBB] 已创建验证器并透传 max_det={getattr(validator.args, 'max_det', None)}")

    stats = validator(model=str(weights))  # 执行验证，返回字典：precision/recall/mAP50/mAP50-95 等

    # 汇总每个类别的详细指标（precision/recall/AP50/AP50-95），并写入结果文件
    # 说明：
    # - validator.metrics.box 提供分类维度上的 p、r、ap50、ap 列表
    # - validator.names 为类别索引到名称的映射字典
    names = getattr(validator, "names", {})
    nc = getattr(validator, "nc", len(names) if isinstance(names, dict) else 0)
    per_class = []
    if nc:
        for i in range(nc):
            try:
                p_i, r_i, ap50_i, ap_i = validator.metrics.class_result(i)
            except Exception:
                p_i, r_i, ap50_i, ap_i = 0.0, 0.0, 0.0, 0.0
            cname = names.get(i, str(i)) if isinstance(names, dict) else str(i)
            per_class.append(
                {
                    "id": i,
                    "name": cname,
                    "precision": float(p_i),
                    "recall": float(r_i),
                    "ap50": float(ap50_i),
                    "ap": float(ap_i),
                }
            )

    enriched = dict(stats)
    enriched["names"] = names
    enriched["classes"] = per_class

    # 保存 JSON 结果（用于后续绘图脚本读取），到 result/<name>/<name>.json
    json_file = out_dir / f"{name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    # 同步保存 CSV 版本（便于表格使用），到 result/<name>/<name>.csv
    csv_file = out_dir / f"{name}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # 总体指标
        w.writerow(["metric", "value"])  # 表头
        for k, v in stats.items():
            w.writerow([k, v])
        # 空行分隔
        w.writerow([])
        # 每类详细指标
        w.writerow(["class", "precision", "recall", "ap50", "ap"])  # 类别指标表头
        for item in per_class:
            w.writerow([item["name"], item["precision"], item["recall"], item["ap50"], item["ap"]])

    # 验证器将在 `result/<run_name>/` 目录下生成 `predictions.json`
    return out_dir



def main() -> None:
    """
    主入口：
    1) 自动/手动指定权重，在测试集上进行评估；
    2) 输出总体与每类指标到指定目录的 JSON/CSV；
    3) 不进行任何推理图片的绘制或保存。
    """
    # 命令行参数（默认 GPU 0，支持测试集比例与结果路径/命名配置）
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="运行设备：cpu 或 GPU 索引，如 0/1")
    parser.add_argument("--weights", type=str, default="", help="权重文件或包含 weights/ 的目录（可选）")
    parser.add_argument("--model-name", type=str, default=DEFAULT_RUN_NAME, help="模型名称（用于结果文件命名）")
    parser.add_argument("--result-dir", type=str, default=str(DEFAULT_RESULT_DIR), help="结果输出目录")
    parser.add_argument("--test-ratio", type=float, default=1.0, help="测试集比例 (0-1]，默认 1.0 全量")
    parser.add_argument("--test-aug", action="store_true", help="启用测试时增强（多尺度/翻转等）")
    parser.add_argument("--iou", type=float, default=IOU_THRES, help="NMS 的 IOU 阈值，默认 0.75 提升召回")
    parser.add_argument("--max-det", type=int, default=MAX_DET, help="每图最大检测数，默认 500")
    parser.add_argument("--data-cfg", type=str, default=str(DATA_CFG), help="数据集 YAML 路径")
    args_ns = parser.parse_args()

    weights_input = Path(args_ns.weights) if args_ns.weights else _find_weights(WEIGHTS_DIR)
    # 若传入的是目录，则自动查找其中的 best/last.pt
    weights = weights_input if weights_input.is_file() else _find_weights(weights_input)

    _ = run_eval(
        weights=weights,
        device=args_ns.device,
        data_cfg=Path(args_ns.data_cfg),
        result_dir=Path(args_ns.result_dir),
        run_name=args_ns.model_name,
        test_ratio=float(args_ns.test_ratio),
        test_aug=bool(args_ns.test_aug),
        iou=float(args_ns.iou),
        max_det=int(args_ns.max_det),
    )


if __name__ == "__main__":
    main()
