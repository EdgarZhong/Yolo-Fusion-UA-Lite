from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIRS = ["van:car", "freight_car:truck", "car:bus"]
DEFAULT_N_COS_PAIRS = 10000


def _parse_pairs(pairs: list[str]) -> list[tuple[str, str]]:
    out = []
    for p in pairs:
        if ":" not in p:
            raise ValueError(f"class pair 格式错误: {p}")
        a, b = p.split(":", 1)
        out.append((a.strip(), b.strip()))
    return out


def _safe_float(v: float | np.floating | None) -> float | None:
    if v is None:
        return None
    x = float(v)
    if np.isnan(x) or np.isinf(x):
        return None
    return x


def _round_or_none(v: float | None, digits: int = 6) -> float | None:
    if v is None:
        return None
    return round(float(v), digits)


def _sample_cosine_intra(feat: np.ndarray, n_pairs: int, rng: np.random.Generator) -> tuple[float | None, float | None]:
    n = int(len(feat))
    if n < 2:
        return None, None

    norms = np.linalg.norm(feat, axis=1)
    valid = norms > 1e-12
    if int(np.sum(valid)) < 2:
        return None, None

    feat_unit = feat[valid] / norms[valid][:, None]
    n_valid = int(len(feat_unit))
    sims_all: list[np.ndarray] = []
    remain = int(n_pairs)

    while remain > 0:
        draw = max(remain * 2, 32)
        idx_i = rng.integers(0, n_valid, size=draw)
        idx_j = rng.integers(0, n_valid, size=draw)
        mask = idx_i != idx_j
        if not np.any(mask):
            continue
        idx_i = idx_i[mask][:remain]
        idx_j = idx_j[mask][:remain]
        if len(idx_i) == 0:
            continue
        sims = np.sum(feat_unit[idx_i] * feat_unit[idx_j], axis=1)
        sims_all.append(sims.astype(np.float64, copy=False))
        remain -= int(len(idx_i))

    sims_cat = np.concatenate(sims_all, axis=0)
    return _safe_float(np.mean(sims_cat)), _safe_float(np.std(sims_cat, ddof=0))


def _class_metrics(feat: np.ndarray, n_pairs: int, rng: np.random.Generator) -> dict:
    if feat.ndim != 2 or len(feat) == 0:
        return {
            "count": int(len(feat)),
            "intra_var": None,
            "cos_intra_mean": None,
            "cos_intra_std": None,
            "norm_mean": None,
            "norm_std": None,
            "centroid": None,
        }

    intra_var = _safe_float(np.mean(np.var(feat, axis=0)))
    cos_mean, cos_std = _sample_cosine_intra(feat, n_pairs=n_pairs, rng=rng)
    norms = np.linalg.norm(feat, axis=1)
    centroid = np.mean(feat, axis=0)
    return {
        "count": int(len(feat)),
        "intra_var": intra_var,
        "cos_intra_mean": cos_mean,
        "cos_intra_std": cos_std,
        "norm_mean": _safe_float(np.mean(norms)),
        "norm_std": _safe_float(np.std(norms, ddof=0)),
        "centroid": centroid,
    }


def _pair_metrics(
    feat: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    cls_a: str,
    cls_b: str,
    seed: int,
    n_cos_pairs: int,
) -> dict:
    if cls_a not in class_names or cls_b not in class_names:
        raise ValueError(f"未在 class_names 中找到类别: {cls_a}, {cls_b}")

    id_a = class_names.index(cls_a)
    id_b = class_names.index(cls_b)
    feat_a = feat[labels == id_a]
    feat_b = feat[labels == id_b]

    rng = np.random.default_rng(seed)
    stats_a = _class_metrics(feat_a, n_pairs=n_cos_pairs, rng=rng)
    stats_b = _class_metrics(feat_b, n_pairs=n_cos_pairs, rng=rng)

    inter_dist_l2 = None
    inter_dist_cos = None
    snr = None

    centroid_a = stats_a["centroid"]
    centroid_b = stats_b["centroid"]
    if centroid_a is not None and centroid_b is not None:
        inter_dist_l2 = _safe_float(np.linalg.norm(centroid_a - centroid_b))
        denom = (np.linalg.norm(centroid_a) * np.linalg.norm(centroid_b)) + 1e-12
        inter_dist_cos = _safe_float(1.0 - float(np.dot(centroid_a, centroid_b) / denom))
        intra_a = stats_a["intra_var"]
        intra_b = stats_b["intra_var"]
        if intra_a is not None and intra_b is not None:
            snr = _safe_float(inter_dist_l2 / (np.sqrt(intra_a + intra_b) + 1e-12))

    return {
        "class_pair": f"{cls_a}:{cls_b}",
        "class_a": cls_a,
        "class_b": cls_b,
        "n_a": int(stats_a["count"]),
        "n_b": int(stats_b["count"]),
        "intra_var_a": _round_or_none(stats_a["intra_var"]),
        "intra_var_b": _round_or_none(stats_b["intra_var"]),
        "cos_intra_a_mean": _round_or_none(stats_a["cos_intra_mean"]),
        "cos_intra_a_std": _round_or_none(stats_a["cos_intra_std"]),
        "cos_intra_b_mean": _round_or_none(stats_b["cos_intra_mean"]),
        "cos_intra_b_std": _round_or_none(stats_b["cos_intra_std"]),
        "inter_dist_l2": _round_or_none(inter_dist_l2),
        "inter_dist_cos": _round_or_none(inter_dist_cos),
        "discriminative_snr": _round_or_none(snr),
        "norm_mean_a": _round_or_none(stats_a["norm_mean"]),
        "norm_std_a": _round_or_none(stats_a["norm_std"]),
        "norm_mean_b": _round_or_none(stats_b["norm_mean"]),
        "norm_std_b": _round_or_none(stats_b["norm_std"]),
    }


def run_feature_compactness(
    feature_file: Path,
    class_pairs: list[tuple[str, str]],
    seed: int,
    n_cos_pairs: int,
    out_json: Path,
) -> Path:
    data = np.load(feature_file, allow_pickle=True)
    class_names = [str(x) for x in data["class_names"].tolist()]
    labels = data["labels"].astype(np.int64)
    feature_keys = sorted([k for k in data.files if k.startswith("feat_")])
    if not feature_keys:
        raise RuntimeError(f"未在 npz 中找到特征字段: {feature_file}")

    rows: list[dict] = []
    for feature_key in feature_keys:
        feat = data[feature_key]
        if feat.ndim != 2 or feat.shape[0] != labels.shape[0]:
            continue
        for cls_a, cls_b in class_pairs:
            row = _pair_metrics(
                feat=feat,
                labels=labels,
                class_names=class_names,
                cls_a=cls_a,
                cls_b=cls_b,
                seed=seed,
                n_cos_pairs=n_cos_pairs,
            )
            row["feature_key"] = feature_key
            rows.append(row)
            print(
                f"[Compactness] {feature_key} | {cls_a}:{cls_b} | "
                f"intra={row['intra_var_a']}/{row['intra_var_b']} | "
                f"snr={row['discriminative_snr']} | "
                f"cos={row['cos_intra_a_mean']}/{row['cos_intra_b_mean']}"
            )

    payload = {
        "time": datetime.now().isoformat(),
        "feature_file": str(feature_file),
        "seed": int(seed),
        "n_cos_pairs": int(n_cos_pairs),
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = out_json.with_suffix(".csv")
    fieldnames = [
        "feature_key",
        "class_pair",
        "n_a",
        "n_b",
        "intra_var_a",
        "intra_var_b",
        "cos_intra_a_mean",
        "cos_intra_a_std",
        "cos_intra_b_mean",
        "cos_intra_b_std",
        "inter_dist_l2",
        "inter_dist_cos",
        "discriminative_snr",
        "norm_mean_a",
        "norm_std_a",
        "norm_mean_b",
        "norm_std_b",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--class-pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cos-pairs", type=int, default=DEFAULT_N_COS_PAIRS)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "reports" / "feature_compactness_results.json"),
    )
    args = parser.parse_args()

    out = run_feature_compactness(
        feature_file=Path(args.features).resolve(),
        class_pairs=_parse_pairs(list(args.class_pairs)),
        seed=int(args.seed),
        n_cos_pairs=int(args.n_cos_pairs),
        out_json=Path(args.output).resolve(),
    )
    print(f"[Diagnostic][Compactness] done: {out}")


if __name__ == "__main__":
    main()
