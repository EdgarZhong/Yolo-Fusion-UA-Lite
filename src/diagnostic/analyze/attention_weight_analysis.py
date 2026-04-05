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
    out = np.concatenate([sel0, sel1], axis=0)
    rng.shuffle(out)
    return out


def _fisher_per_channel(x: np.ndarray, y: np.ndarray, class0: int, class1: int) -> np.ndarray:
    x0 = x[y == class0]
    x1 = x[y == class1]
    mu0 = np.mean(x0, axis=0)
    mu1 = np.mean(x1, axis=0)
    var0 = np.var(x0, axis=0)
    var1 = np.var(x1, axis=0)
    return (mu0 - mu1) ** 2 / (var0 + var1 + 1e-12)


def _approx_attn(post: np.ndarray, pre: np.ndarray, eps: float) -> np.ndarray:
    safe = np.where(np.abs(pre) < eps, np.sign(pre) * eps, pre)
    safe = np.where(safe == 0.0, eps, safe)
    return post / safe


def run_analysis(
    feature_file: Path,
    class_pairs: list[tuple[str, str]],
    seed: int,
    feature_key: str,
    eps: float,
    out_json: Path,
) -> Path:
    data = np.load(feature_file, allow_pickle=True)
    class_names = [str(x) for x in data["class_names"].tolist()]
    labels = data["labels"].astype(np.int64)
    image_ids = np.asarray(data["image_ids"]).astype(str)

    required = [feature_key, "feat_pre_attn_rgb", "feat_pre_attn_ir", "feat_post_attn_rgb", "feat_post_attn_ir"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(f"npz 缺少近似注意力分析所需字段: {missing}")

    feat = data[feature_key]
    pre_rgb = data["feat_pre_attn_rgb"]
    pre_ir = data["feat_pre_attn_ir"]
    post_rgb = data["feat_post_attn_rgb"]
    post_ir = data["feat_post_attn_ir"]

    approx_rgb = _approx_attn(post_rgb, pre_rgb, eps=eps)
    approx_ir = _approx_attn(post_ir, pre_ir, eps=eps)

    rows = []
    for a, b in class_pairs:
        if a not in class_names or b not in class_names:
            continue
        ia = class_names.index(a)
        ib = class_names.index(b)
        mask = np.logical_or(labels == ia, labels == ib)
        x_ab = feat[mask]
        y_ab = labels[mask]
        im_ab = image_ids[mask]
        rgb_ab = approx_rgb[mask]
        ir_ab = approx_ir[mask]
        if len(np.unique(y_ab)) < 2 or len(y_ab) < 20:
            continue

        balanced = _balanced_indices(y_ab, seed=seed)
        x_bal = x_ab[balanced]
        y_bal = y_ab[balanced]
        im_bal = im_ab[balanced]
        fisher = _fisher_per_channel(x_bal, y_bal, ia, ib)
        target_images = np.unique(im_bal[y_bal == ia])

        for src, attn_ab in [("rgb", rgb_ab), ("ir", ir_ab)]:
            attn_mask = np.isin(im_ab, target_images)
            if not np.any(attn_mask):
                continue
            attn_sel = attn_ab[attn_mask]
            finite_ratio = float(np.mean(np.isfinite(attn_sel)))
            mean_attn = np.nanmean(attn_sel, axis=0)
            dim = min(len(fisher), len(mean_attn))
            fisher_v = fisher[:dim]
            attn_v = mean_attn[:dim]
            valid = np.isfinite(fisher_v) & np.isfinite(attn_v)
            if int(np.sum(valid)) < 3:
                continue
            r, p = pearsonr(fisher_v[valid], attn_v[valid])
            out_of_unit_ratio = float(np.mean(np.logical_or(attn_sel < 0.0, attn_sel > 1.0)))
            rows.append(
                {
                    "feature_key": feature_key,
                    "class_pair": f"{a}:{b}",
                    "attn_source": src,
                    "pearson_r": float(r),
                    "p_value": float(p),
                    "n_objects": int(len(y_bal)),
                    "n_images_attn": int(len(target_images)),
                    "finite_ratio": finite_ratio,
                    "out_of_unit_ratio": out_of_unit_ratio,
                    "approx_method": "output/(input+eps)",
                }
            )

    payload = {
        "time": datetime.now().isoformat(),
        "feature_file": str(feature_file),
        "seed": int(seed),
        "feature_key": feature_key,
        "eps": float(eps),
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = out_json.with_suffix(".csv")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "feature_key",
                "class_pair",
                "attn_source",
                "pearson_r",
                "p_value",
                "n_objects",
                "n_images_attn",
                "finite_ratio",
                "out_of_unit_ratio",
                "approx_method",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--class-pairs", nargs="+", default=["van:car", "freight_car:truck", "car:bus"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-key", type=str, default="feat_fused_p3")
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "reports" / "attention_weight_analysis.json"),
    )
    args = parser.parse_args()

    out = run_analysis(
        feature_file=Path(args.features).resolve(),
        class_pairs=_parse_pairs(args.class_pairs),
        seed=int(args.seed),
        feature_key=str(args.feature_key),
        eps=float(args.eps),
        out_json=Path(args.output).resolve(),
    )
    print(f"[Diagnostic][AttentionApprox] done: {out}")


if __name__ == "__main__":
    main()
