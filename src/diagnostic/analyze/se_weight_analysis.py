from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[3]


def _parse_pairs(pairs: list[str]) -> list[tuple[str, str]]:
    out = []
    for p in pairs:
        if ":" not in p:
            raise ValueError(f"class pair 格式错误: {p}")
        a, b = p.split(":", 1)
        out.append((a.strip(), b.strip()))
    return out


def _balanced_indices(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uniq = np.unique(y)
    if len(uniq) != 2:
        raise ValueError("当前实现仅支持二分类 pair")
    idx0 = np.where(y == uniq[0])[0]
    idx1 = np.where(y == uniq[1])[0]
    n = min(len(idx0), len(idx1))
    sel0 = rng.choice(idx0, size=n, replace=False)
    sel1 = rng.choice(idx1, size=n, replace=False)
    all_idx = np.concatenate([sel0, sel1], axis=0)
    rng.shuffle(all_idx)
    return all_idx


def _fisher_per_channel(x: np.ndarray, y: np.ndarray, class0: int, class1: int) -> np.ndarray:
    x0 = x[y == class0]
    x1 = x[y == class1]
    mu0 = np.mean(x0, axis=0)
    mu1 = np.mean(x1, axis=0)
    var0 = np.var(x0, axis=0)
    var1 = np.var(x1, axis=0)
    return (mu0 - mu1) ** 2 / (var0 + var1 + 1e-12)


def _resolve_attn_fields(data: np.lib.npyio.NpzFile) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if "attn_weights_rgb" in data.files and "attn_image_ids" in data.files:
        fields["rgb"] = (data["attn_weights_rgb"], np.asarray(data["attn_image_ids"]).astype(str))
    if "attn_weights_ir" in data.files and "attn_image_ids" in data.files:
        fields["ir"] = (data["attn_weights_ir"], np.asarray(data["attn_image_ids"]).astype(str))
    if "attn_weights" in data.files and "attn_image_ids" in data.files:
        fields["shared"] = (data["attn_weights"], np.asarray(data["attn_image_ids"]).astype(str))
    return fields


def run_analysis(
    feature_file: Path,
    class_pairs: list[tuple[str, str]],
    seed: int,
    out_json: Path,
) -> Path:
    data = np.load(feature_file, allow_pickle=True)
    class_names = [str(x) for x in data["class_names"].tolist()]
    labels = data["labels"].astype(np.int64)
    image_ids = np.asarray(data["image_ids"]).astype(str)
    attn_fields = _resolve_attn_fields(data)
    if not attn_fields:
        raise RuntimeError("npz 中未找到注意力权重字段（attn_weights 或 attn_weights_rgb）")

    feature_keys = [k for k in ["feat_pre_attn_rgb", "feat_pre_attn_ir", "feat_fused_p3"] if k in data.files]
    if not feature_keys:
        raise RuntimeError("npz 中未找到可用于机制分析的特征字段（feat_pre_attn_* / feat_fused_p3）")

    rows = []
    for feat_key in feature_keys:
        x = data[feat_key]
        if x.ndim != 2 or x.shape[0] != labels.shape[0]:
            continue
        for a, b in class_pairs:
            if a not in class_names or b not in class_names:
                continue
            ia = class_names.index(a)
            ib = class_names.index(b)

            mask = np.logical_or(labels == ia, labels == ib)
            x_ab = x[mask]
            y_ab = labels[mask]
            im_ab = image_ids[mask]
            if len(np.unique(y_ab)) < 2 or len(y_ab) < 20:
                continue

            balanced = _balanced_indices(y_ab, seed=seed)
            x_bal = x_ab[balanced]
            y_bal = y_ab[balanced]
            im_bal = im_ab[balanced]

            fisher = _fisher_per_channel(x_bal, y_bal, ia, ib)
            target_images = np.unique(im_bal[y_bal == ia])
            if feat_key == "feat_pre_attn_rgb":
                sources = ["rgb"] if "rgb" in attn_fields else ["shared"]
            elif feat_key == "feat_pre_attn_ir":
                sources = ["ir"] if "ir" in attn_fields else ["shared"]
            else:
                if "rgb" in attn_fields and "ir" in attn_fields:
                    sources = ["rgb", "ir"]
                else:
                    sources = ["shared"]
            for src in sources:
                if src not in attn_fields:
                    continue
                attn_weights, attn_image_ids = attn_fields[src]
                attn_mask = np.isin(attn_image_ids, target_images)
                if not np.any(attn_mask):
                    continue
                attn_mean = np.mean(attn_weights[attn_mask], axis=0)
                dim = min(len(fisher), len(attn_mean))
                fisher_v = fisher[:dim]
                attn_v = attn_mean[:dim]
                r, p = pearsonr(fisher_v, attn_v)
                rows.append(
                    {
                        "feature_key": feat_key,
                        "class_pair": f"{a}:{b}",
                        "attn_source": src,
                        "pearson_r": float(r),
                        "p_value": float(p),
                        "n_objects": int(len(y_bal)),
                        "n_images_attn": int(np.sum(attn_mask)),
                    }
                )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": datetime.now().isoformat(),
        "feature_file": str(feature_file),
        "seed": int(seed),
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = out_json.with_suffix(".csv")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["feature_key", "class_pair", "attn_source", "pearson_r", "p_value", "n_objects", "n_images_attn"],
        )
        w.writeheader()
        w.writerows(rows)
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--class-pairs", nargs="+", default=["van:car", "freight_car:truck", "car:bus"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "reports" / "se_weight_analysis.json"),
    )
    args = parser.parse_args()

    out = run_analysis(
        feature_file=Path(args.features).resolve(),
        class_pairs=_parse_pairs(args.class_pairs),
        seed=int(args.seed),
        out_json=Path(args.output).resolve(),
    )
    print(f"[Diagnostic][SE-Analysis] done: {out}")


if __name__ == "__main__":
    main()
