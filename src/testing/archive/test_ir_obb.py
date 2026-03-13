from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import argparse

import torch

ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBValidator  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.data import build_yolo_dataset  # noqa: E402

DATA_CFG = ROOT / "src/cfg/datasets/ir_obb_dronevehicle.yaml"
WEIGHTS_DIR = ROOT / "models/IR-YOLOv8n"
DEFAULT_RESULT_DIR = ROOT / "result"
DEFAULT_RUN_NAME = "IR-YOLOv8n"

IMG_SIZE = 640
BATCH = 16
CONF_THRES = 0.25
IOU_THRES = 0.75
MAX_DET = 300


def _find_weights(base: Path) -> Path:
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
    result_dir: Path | None = None,
    run_name: str | None = None,
    test_ratio: float = 1.0,
    test_aug: bool = True,
    iou: float = IOU_THRES,
) -> Path:
    result_root = Path(result_dir) if result_dir else DEFAULT_RESULT_DIR
    result_root.mkdir(parents=True, exist_ok=True)
    name = run_name or DEFAULT_RUN_NAME

    workers = 0 if str(device).lower() == "cpu" else 2

    args = dict(
        task="obb",
        model=str(weights),
        data=str(DATA_CFG),
        split="test",
        imgsz=IMG_SIZE,
        rect=True,
        batch=BATCH,
        workers=workers,
        device=device,
        plots=False,
        save_json=True,
        conf=CONF_THRES,
        iou=float(iou),
        augment=bool(test_aug),
        max_det=MAX_DET,
        project=str(result_root),
        name=name,
    )

    if isinstance(test_ratio, float) and 0.0 < test_ratio < 1.0:
        data = check_det_dataset(str(DATA_CFG))
        cfg = get_cfg(overrides=dict(task="obb", imgsz=IMG_SIZE, rect=True))
        base_ds = build_yolo_dataset(cfg, data.get("test"), BATCH, data, mode="val")
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
    else:
        validator = OBBValidator(args=args)

    stats = validator(model=str(weights))

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

    json_file = result_root / f"{name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    csv_file = result_root / f"{name}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in stats.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["class", "precision", "recall", "ap50", "ap"])
        for item in per_class:
            w.writerow([item["name"], item["precision"], item["recall"], item["ap50"], item["ap"]])

    return result_root / name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--model-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--result-dir", type=str, default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--test-ratio", type=float, default=1.0)
    parser.add_argument("--test-aug", action="store_true")
    parser.add_argument("--iou", type=float, default=IOU_THRES)
    args_ns = parser.parse_args()

    weights_input = Path(args_ns.weights) if args_ns.weights else _find_weights(WEIGHTS_DIR)
    weights = weights_input if weights_input.is_file() else _find_weights(weights_input)

    _ = run_eval(
        weights=weights,
        device=args_ns.device,
        result_dir=Path(args_ns.result_dir),
        run_name=args_ns.model_name,
        test_ratio=float(args_ns.test_ratio),
        test_aug=bool(args_ns.test_aug),
        iou=float(args_ns.iou),
    )


if __name__ == "__main__":
    main()
