from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBValidator  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.data import build_yolo_dataset  # noqa: E402
from ultralytics.utils import ops  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402


DEFAULT_DATA_CFG = ROOT / "src/cfg/datasets/dual_obb_dronevehicle.yaml"
DEFAULT_RESULT_DIR = ROOT / "result_hbb"
IMG_SIZE = 640
BATCH = 16
CONF_THRES = 0.25
IOU_THRES = 0.75
MAX_DET = 500


def _find_weights(base: Path) -> Path:
    wdir = base / "weights"
    for p in (wdir / "best.pt", wdir / "last.pt"):
        if p.exists():
            return p
    other = next(iter(wdir.glob("*.pt")), None)
    if other is None:
        raise FileNotFoundError(f"未找到权重文件：{wdir}")
    return other


class HBBFromOBBValidator(OBBValidator):
    @staticmethod
    def _obb_xywhr_to_hbb_xyxy(xywhr: torch.Tensor) -> torch.Tensor:
        if xywhr.numel() == 0:
            return torch.zeros((0, 4), device=xywhr.device, dtype=xywhr.dtype)
        poly = ops.xywhr2xyxyxyxy(xywhr).view(-1, 4, 2)
        x_min = poly[..., 0].min(dim=1).values
        y_min = poly[..., 1].min(dim=1).values
        x_max = poly[..., 0].max(dim=1).values
        y_max = poly[..., 1].max(dim=1).values
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)

    @staticmethod
    def _class_aware_hbb_nms(boxes: torch.Tensor, scores: torch.Tensor, cls: torch.Tensor, iou_thres: float, max_det: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.zeros((0,), device=boxes.device, dtype=torch.long)
        keep_all = []
        for c in cls.unique():
            inds = torch.where(cls == c)[0]
            b = boxes[inds]
            s = scores[inds]
            order = torch.argsort(s, descending=True)
            while order.numel() > 0:
                i = order[0]
                keep_all.append(inds[i])
                if order.numel() == 1:
                    break
                ious = box_iou(b[i].unsqueeze(0), b[order[1:]]).squeeze(0)
                order = order[1:][ious <= float(iou_thres)]
        if not keep_all:
            return torch.zeros((0,), device=boxes.device, dtype=torch.long)
        keep = torch.stack(keep_all)
        keep = keep[torch.argsort(scores[keep], descending=True)]
        return keep[: int(max_det)]

    def postprocess(self, preds):
        protocol = str(getattr(self, "hbb_protocol", "strict")).lower()
        preds_obb = ops.non_max_suppression(
            preds,
            float(getattr(self.args, "conf", 0.25)),
            0.999 if protocol == "strict" else float(self.args.iou),
            labels=self.lb,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=max(int(self.args.max_det) * 20, int(self.args.max_det)),
            rotated=True,
            max_time_img=0.2,
        )
        if protocol != "strict":
            return preds_obb
        out = []
        for pred in preds_obb:
            if pred is None or len(pred) == 0:
                out.append(pred)
                continue
            rbox = torch.cat([pred[:, :4], pred[:, -1:]], dim=1)
            hbb = self._obb_xywhr_to_hbb_xyxy(rbox)
            keep = self._class_aware_hbb_nms(hbb, pred[:, 4], pred[:, 5], float(self.args.iou), int(self.args.max_det))
            out.append(pred[keep] if len(keep) else pred[:0])
        return out

    def _prepare_batch(self, si, batch):
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx].clone()
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])
            ops.scale_boxes(imgsz, bbox[..., :4], ori_shape, ratio_pad=ratio_pad, xywh=True)
            bbox = self._obb_xywhr_to_hbb_xyxy(bbox)
        else:
            bbox = torch.zeros((0, 4), device=self.device)
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        predn = pred.clone()
        ops.scale_boxes(pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True)
        pred_xyxy = self._obb_xywhr_to_hbb_xyxy(torch.cat([predn[:, :4], predn[:, -1:]], dim=1))
        return torch.cat([pred_xyxy, predn[:, 4:6]], dim=1)

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)


def run_eval(
    weights: Path,
    run_name: str,
    device: str = "0",
    data_cfg: Path = DEFAULT_DATA_CFG,
    hbb_protocol: str = "strict",
    result_dir: Path | None = None,
    test_ratio: float = 1.0,
    test_aug: bool = False,
    conf: float = CONF_THRES,
    iou: float = IOU_THRES,
    max_det: int = MAX_DET,
) -> Path:
    result_root = Path(result_dir) if result_dir else DEFAULT_RESULT_DIR
    result_root.mkdir(parents=True, exist_ok=True)
    out_dir = result_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = 0 if str(device).lower() == "cpu" else 2
    args = dict(
        task="obb",
        model=str(weights),
        data=str(data_cfg),
        split="test",
        imgsz=IMG_SIZE,
        rect=True,
        batch=BATCH,
        workers=workers,
        device=device,
        plots=False,
        save_json=False,
        conf=float(conf),
        iou=float(iou),
        augment=bool(test_aug),
        max_det=int(max_det),
        project=str(result_root),
        name=run_name,
    )

    if isinstance(test_ratio, float) and 0.0 < test_ratio < 1.0:
        data = check_det_dataset(str(data_cfg))
        cfg = get_cfg(overrides=dict(task="obb", imgsz=IMG_SIZE, rect=True))
        base_ds = build_yolo_dataset(cfg, data.get("test"), BATCH, data, mode="val")
        from torch.utils.data import DataLoader, Subset

        count = max(1, int(len(base_ds) * test_ratio))
        subset = Subset(base_ds, list(range(count)))
        loader = DataLoader(
            dataset=subset,
            batch_size=min(BATCH, len(subset)),
            shuffle=False,
            num_workers=workers,
            collate_fn=getattr(base_ds, "collate_fn", None),
            pin_memory=False,
        )
        validator = HBBFromOBBValidator(dataloader=loader, args=args)
    else:
        validator = HBBFromOBBValidator(args=args)
    validator.hbb_protocol = str(hbb_protocol).lower()

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
    enriched["eval_mode"] = "HBB_from_OBB"
    enriched["hbb_protocol"] = str(hbb_protocol)
    enriched["weights"] = str(weights)
    enriched["params"] = {
        "imgsz": IMG_SIZE,
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
        "split": "test",
        "batch": BATCH,
        "augment": bool(test_aug),
    }
    enriched["names"] = names
    enriched["classes"] = per_class

    json_file = out_dir / f"{run_name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    csv_file = out_dir / f"{run_name}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in stats.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["class", "precision", "recall", "ap50", "ap"])
        for item in per_class:
            w.writerow([item["name"], item["precision"], item["recall"], item["ap50"], item["ap"]])

    return out_dir


def _parse_float_list(text: str) -> list[float]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    return [float(x) for x in vals]


def _parse_int_list(text: str) -> list[int]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    return [int(float(x)) for x in vals]


def run_sweep(
    weights: Path,
    run_name: str,
    device: str,
    data_cfg: Path,
    hbb_protocol: str,
    result_dir: Path,
    test_ratio: float,
    test_aug: bool,
    confs: list[float],
    ious: list[float],
    max_dets: list[int],
    optimize_key: str,
) -> Path:
    result_root = Path(result_dir)
    result_root.mkdir(parents=True, exist_ok=True)
    sweep_root = result_root / run_name
    sweep_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    best = None
    for conf, iou, max_det in itertools.product(confs, ious, max_dets):
        tag = f"c{conf:.3f}_i{iou:.3f}_m{int(max_det)}".replace(".", "p")
        trial_name = f"{run_name}_{tag}"
        out_dir = run_eval(
            weights=weights,
            run_name=trial_name,
            device=device,
            data_cfg=data_cfg,
            hbb_protocol=hbb_protocol,
            result_dir=result_root,
            test_ratio=test_ratio,
            test_aug=test_aug,
            conf=conf,
            iou=iou,
            max_det=max_det,
        )
        trial_json = out_dir / f"{trial_name}.json"
        trial_data = json.loads(trial_json.read_text(encoding="utf-8"))
        metric = float(trial_data.get(optimize_key, 0.0))
        row = {
            "trial_name": trial_name,
            "conf": float(conf),
            "iou": float(iou),
            "max_det": int(max_det),
            "metric": metric,
            "json_path": str(trial_json),
        }
        rows.append(row)
        if best is None or metric > best["metric"]:
            best = row

    rows = sorted(rows, key=lambda x: x["metric"], reverse=True)
    sweep_csv = sweep_root / f"{run_name}_sweep.csv"
    with sweep_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial_name", "conf", "iou", "max_det", "metric", "json_path"])
        w.writeheader()
        w.writerows(rows)

    best_json = sweep_root / f"{run_name}_best_params.json"
    best_payload = {
        "optimize_key": optimize_key,
        "best": best,
        "top5": rows[:5],
    }
    best_json.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[HBB-Sweep] 完成: {sweep_root}")
    print(f"[HBB-Sweep] 最优: conf={best['conf']}, iou={best['iou']}, max_det={best['max_det']}, {optimize_key}={best['metric']}")
    return best_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--data-cfg", type=str, default=str(DEFAULT_DATA_CFG))
    parser.add_argument("--result-dir", type=str, default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--test-ratio", type=float, default=1.0)
    parser.add_argument("--test-aug", action="store_true")
    parser.add_argument("--hbb-protocol", type=str, choices=["strict", "legacy"], default="strict")
    parser.add_argument("--conf", type=float, default=CONF_THRES)
    parser.add_argument("--iou", type=float, default=IOU_THRES)
    parser.add_argument("--max-det", type=int, default=MAX_DET)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-confs", type=str, default="0.20,0.25,0.30")
    parser.add_argument("--sweep-ious", type=str, default="0.65,0.75")
    parser.add_argument("--sweep-max-dets", type=str, default="300,500")
    parser.add_argument("--optimize-key", type=str, default="metrics/mAP50-95(B)")
    args = parser.parse_args()

    weights_input = Path(args.weights)
    weights = weights_input if weights_input.is_file() else _find_weights(weights_input)
    if bool(args.sweep):
        best_json = run_sweep(
            weights=weights,
            run_name=args.model_name,
            device=args.device,
            data_cfg=Path(args.data_cfg),
            hbb_protocol=str(args.hbb_protocol),
            result_dir=Path(args.result_dir),
            test_ratio=float(args.test_ratio),
            test_aug=bool(args.test_aug),
            confs=_parse_float_list(args.sweep_confs),
            ious=_parse_float_list(args.sweep_ious),
            max_dets=_parse_int_list(args.sweep_max_dets),
            optimize_key=str(args.optimize_key),
        )
        print(f"[HBB-Sweep] best_params_file: {best_json}")
    else:
        out_dir = run_eval(
            weights=weights,
            run_name=args.model_name,
            device=args.device,
            data_cfg=Path(args.data_cfg),
            hbb_protocol=str(args.hbb_protocol),
            result_dir=Path(args.result_dir),
            test_ratio=float(args.test_ratio),
            test_aug=bool(args.test_aug),
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
        )
        print(f"[HBB-Eval] 完成: {out_dir}")


if __name__ == "__main__":
    main()
