from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data import build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils import ops  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


ALLOWED_SEMANTIC_KEYS = {
    "rgb_p3",
    "ir_p3",
    "pre_attn_rgb",
    "pre_attn_ir",
    "fused_p3",
    "attn_weights",
}


@dataclass
class HookSpec:
    name: str
    layer_idx: int | None
    module_path: str | None
    capture: str


def _resolve_module_by_path(root_module: torch.nn.Module, module_path: str) -> torch.nn.Module:
    cur = root_module
    for token in module_path.split("."):
        token = token.strip()
        if token.isdigit():
            cur = cur[int(token)]
        else:
            cur = getattr(cur, token)
    return cur


def _to_numpy_feature(vectors: list[np.ndarray], dtype=np.float32) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype=dtype)
    return np.stack(vectors, axis=0).astype(dtype, copy=False)


def _parse_hook_config(path: Path) -> tuple[str, list[HookSpec], dict]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_name = str(cfg.get("model_name", path.stem))
    hooks = []
    hook_map = cfg.get("hooks", {})
    for k, v in hook_map.items():
        if k not in ALLOWED_SEMANTIC_KEYS:
            raise ValueError(f"hook key 不在冻结语义集合中: {k}")
        layer_idx = v.get("layer_idx", None)
        module_path = v.get("module_path", None)
        capture = str(v.get("capture", "output")).lower()
        if capture not in {"output", "input", "weights", "weights_pair"}:
            raise ValueError(f"capture 类型非法: {capture}")
        if layer_idx is None and module_path is None:
            raise ValueError(f"hook {k} 需要 layer_idx 或 module_path")
        hooks.append(
            HookSpec(
                name=k,
                layer_idx=int(layer_idx) if layer_idx is not None else None,
                module_path=str(module_path) if module_path is not None else None,
                capture=capture,
            )
        )
    if not hooks:
        raise ValueError("hook 配置为空")
    return model_name, hooks, cfg


def _compute_image_brightness(image_path: str, cache: dict[str, float]) -> float:
    if image_path in cache:
        return cache[image_path]
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        cache[image_path] = float("nan")
        return cache[image_path]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    cache[image_path] = float(np.mean(lab[:, :, 0]))
    return cache[image_path]


def _resolve_rgb_path(raw_im_file: str) -> str:
    text = str(raw_im_file)
    if "|" in text:
        left = text.split("|", 1)[0].strip()
        return left if left else text
    return text


def _obb_to_aabb_xyxy(obb_xywhr: torch.Tensor) -> torch.Tensor:
    if obb_xywhr.numel() == 0:
        return torch.zeros((0, 4), device=obb_xywhr.device, dtype=obb_xywhr.dtype)
    poly = ops.xywhr2xyxyxyxy(obb_xywhr).view(-1, 4, 2)
    x_min = poly[..., 0].min(dim=1).values
    y_min = poly[..., 1].min(dim=1).values
    x_max = poly[..., 0].max(dim=1).values
    y_max = poly[..., 1].max(dim=1).values
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


def _map_roi_to_feature(
    xyxy: torch.Tensor,
    ori_w: int,
    ori_h: int,
    feat_w: int,
    feat_h: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    x0_f = int(round(x0 * feat_w / max(1, ori_w)))
    y0_f = int(round(y0 * feat_h / max(1, ori_h)))
    x1_f = int(round(x1 * feat_w / max(1, ori_w)))
    y1_f = int(round(y1 * feat_h / max(1, ori_h)))

    x0_f = max(0, min(feat_w - 1, x0_f))
    x1_f = max(0, min(feat_w - 1, x1_f))
    y0_f = max(0, min(feat_h - 1, y0_f))
    y1_f = max(0, min(feat_h - 1, y1_f))

    if x1_f < x0_f:
        x0_f, x1_f = x1_f, x0_f
    if y1_f < y0_f:
        y0_f, y1_f = y1_f, y0_f

    width = x1_f - x0_f + 1
    height = y1_f - y0_f + 1
    if width < 2:
        c = (x0_f + x1_f) // 2
        x0_f = max(0, c - 1)
        x1_f = min(feat_w - 1, x0_f + 1)
        if x1_f - x0_f + 1 < 2 and x0_f > 0:
            x0_f -= 1
    if height < 2:
        c = (y0_f + y1_f) // 2
        y0_f = max(0, c - 1)
        y1_f = min(feat_h - 1, y0_f + 1)
        if y1_f - y0_f + 1 < 2 and y0_f > 0:
            y0_f -= 1

    return x0_f, y0_f, x1_f, y1_f


def _capture_weights_from_se(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    w = module.avg(x)
    w = module.fc2(module.act(module.fc1(w)))
    w = module.gate(w)
    return w


def _capture_pair_weights_from_fa(module: torch.nn.Module, x_rgb: torch.Tensor, x_ir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fr = module.inc_rgb(x_rgb)
    fi = module.inc_ir(x_ir)
    wr = _capture_weights_from_se(module.se_rgb, fr)
    wi = _capture_weights_from_se(module.se_ir, fi)
    wr = wr.view(wr.shape[0], wr.shape[1], -1).mean(dim=2)
    wi = wi.view(wi.shape[0], wi.shape[1], -1).mean(dim=2)
    return wr, wi


def run_extract(
    weights: Path,
    data_cfg: Path,
    hook_cfg: Path,
    output_npz: Path,
    device: str,
    split: str,
    imgsz: int,
    batch: int,
    workers: int,
    max_samples: int,
) -> Path:
    model_name, hook_specs, hook_cfg_raw = _parse_hook_config(hook_cfg)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    dev = select_device(device, batch=batch)
    yolo = YOLO(str(weights))
    model = yolo.model.to(dev)
    model.eval()

    data = check_det_dataset(str(data_cfg))
    cfg = get_cfg(overrides=dict(task="obb", imgsz=imgsz, rect=True))
    ds = build_yolo_dataset(cfg, data.get(split), batch, data, mode="val")
    loader = DataLoader(
        dataset=ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers if str(dev) != "cpu" else 0,
        collate_fn=getattr(ds, "collate_fn", None),
        pin_memory=False,
    )

    captured: dict[str, torch.Tensor] = {}
    handles = []
    for spec in hook_specs:
        if spec.layer_idx is not None:
            module = model.model[int(spec.layer_idx)]
        else:
            module = _resolve_module_by_path(model, str(spec.module_path))

        def _hook_fn(m, inp, out, _spec=spec):
            if _spec.capture == "output":
                captured[_spec.name] = out.detach()
            elif _spec.capture == "input":
                captured[_spec.name] = inp[0].detach()
            elif _spec.capture == "weights":
                captured[_spec.name] = _capture_weights_from_se(m, inp[0]).detach()
            else:
                wr, wi = _capture_pair_weights_from_fa(m, inp[0], inp[1])
                captured["attn_weights_rgb"] = wr.detach()
                captured["attn_weights_ir"] = wi.detach()

        handles.append(module.register_forward_hook(_hook_fn))

    feature_pool: dict[str, list[np.ndarray]] = defaultdict(list)
    labels: list[int] = []
    image_ids: list[str] = []
    bbox_areas_px: list[float] = []
    image_brightness: list[float] = []
    attn_weights: list[np.ndarray] = []
    attn_weights_rgb: list[np.ndarray] = []
    attn_weights_ir: list[np.ndarray] = []
    attn_image_ids: list[str] = []
    brightness_cache: dict[str, float] = {}
    raw_names = data.get("names", [])
    if isinstance(raw_names, dict):
        names = [str(raw_names[k]) for k in sorted(raw_names)]
    else:
        names = [str(x) for x in raw_names]

    processed_images = 0
    for batch_data in loader:
        imgs = batch_data["img"].to(dev, non_blocking=True).float()
        if imgs.max() > 1.0:
            imgs = imgs / 255.0

        captured.clear()
        with torch.no_grad():
            _ = model(imgs)

        cur_batch = int(imgs.shape[0])
        img_h, img_w = int(imgs.shape[2]), int(imgs.shape[3])
        batch_idx = batch_data["batch_idx"]
        cls_all = batch_data["cls"].squeeze(-1)
        bbox_all = batch_data["bboxes"].clone()
        bbox_all[..., :4] *= torch.tensor([img_w, img_h, img_w, img_h], dtype=bbox_all.dtype)

        for bi in range(cur_batch):
            processed_images += 1
            if max_samples > 0 and processed_images > max_samples:
                break

            im_file = str(batch_data["im_file"][bi])
            rgb_path = _resolve_rgb_path(im_file)
            ori_shape = batch_data["ori_shape"][bi]
            ori_h, ori_w = int(ori_shape[0]), int(ori_shape[1])
            ratio_pad = batch_data["ratio_pad"][bi]

            obj_mask = batch_idx == bi
            cls_i = cls_all[obj_mask].cpu().numpy().astype(np.int64)
            bbox_i = bbox_all[obj_mask]
            if len(bbox_i):
                ops.scale_boxes((img_h, img_w), bbox_i, (ori_h, ori_w), ratio_pad=ratio_pad, xywh=True)
                aabb_i = _obb_to_aabb_xyxy(bbox_i).cpu()
            else:
                aabb_i = torch.zeros((0, 4))

            bright = _compute_image_brightness(rgb_path, brightness_cache)

            for oi in range(len(cls_i)):
                labels.append(int(cls_i[oi]))
                image_ids.append(Path(rgb_path).name)
                x0, y0, x1, y1 = [float(v) for v in aabb_i[oi].tolist()]
                bbox_areas_px.append(max(0.0, (x1 - x0) * (y1 - y0)))
                image_brightness.append(float(bright))

                for spec in hook_specs:
                    if spec.name == "attn_weights":
                        continue
                    fmap = captured.get(spec.name, None)
                    if fmap is None:
                        continue
                    one = fmap[bi]
                    feat_h, feat_w = int(one.shape[1]), int(one.shape[2])
                    rx0, ry0, rx1, ry1 = _map_roi_to_feature(aabb_i[oi], ori_w, ori_h, feat_w, feat_h)
                    roi = one[:, ry0 : ry1 + 1, rx0 : rx1 + 1]
                    vec = roi.mean(dim=(1, 2)).cpu().numpy().astype(np.float32)
                    feature_pool[f"feat_{spec.name}"].append(vec)

            if "attn_weights" in captured:
                aw = captured["attn_weights"][bi]
                aw = aw.view(aw.shape[0], -1).mean(dim=1).cpu().numpy().astype(np.float32)
                attn_weights.append(aw)
                attn_image_ids.append(Path(rgb_path).name)
            if "attn_weights_rgb" in captured and "attn_weights_ir" in captured:
                awr = captured["attn_weights_rgb"][bi].cpu().numpy().astype(np.float32)
                awi = captured["attn_weights_ir"][bi].cpu().numpy().astype(np.float32)
                attn_weights_rgb.append(awr)
                attn_weights_ir.append(awi)
                if not attn_image_ids or attn_image_ids[-1] != Path(rgb_path).name:
                    attn_image_ids.append(Path(rgb_path).name)

        if max_samples > 0 and processed_images >= max_samples:
            break

    for h in handles:
        h.remove()

    payload: dict[str, np.ndarray | str] = {
        "model_name": np.asarray([model_name], dtype=object),
        "labels": np.asarray(labels, dtype=np.int64),
        "image_ids": np.asarray(image_ids, dtype=object),
        "bbox_areas_px": np.asarray(bbox_areas_px, dtype=np.float32),
        "image_brightness": np.asarray(image_brightness, dtype=np.float32),
        "class_names": np.asarray(names, dtype=object),
    }
    for k, vectors in feature_pool.items():
        payload[k] = _to_numpy_feature(vectors)
    if attn_weights:
        payload["attn_weights"] = _to_numpy_feature(attn_weights)
    if attn_weights_rgb and attn_weights_ir:
        payload["attn_weights_rgb"] = _to_numpy_feature(attn_weights_rgb)
        payload["attn_weights_ir"] = _to_numpy_feature(attn_weights_ir)
    if attn_image_ids:
        payload["attn_image_ids"] = np.asarray(attn_image_ids, dtype=object)

    np.savez(output_npz, **payload)

    config_snapshot = {
        "time": datetime.now().isoformat(),
        "weights": str(weights),
        "data_cfg": str(data_cfg),
        "hook_cfg": str(hook_cfg),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "max_samples": max_samples,
        "device": str(dev),
        "features": sorted([k for k in payload.keys() if str(k).startswith("feat_")]),
        "n_objects": int(len(labels)),
        "n_images": int(processed_images),
        "hook_config_raw": hook_cfg_raw,
    }
    (output_npz.parent / f"{output_npz.stem}.extract_config.json").write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data-cfg", type=str, required=True)
    parser.add_argument("--hook-cfg", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "features" / "m5_fa_concat.npz"),
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    out = run_extract(
        weights=Path(args.weights).resolve(),
        data_cfg=Path(args.data_cfg).resolve(),
        hook_cfg=Path(args.hook_cfg).resolve(),
        output_npz=Path(args.output).resolve(),
        device=str(args.device),
        split=str(args.split),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        max_samples=int(args.max_samples),
    )
    print(f"[Diagnostic][Extract] done: {out}")


if __name__ == "__main__":
    main()
