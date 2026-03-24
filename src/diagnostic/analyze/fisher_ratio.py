from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


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
    denom = var0 + var1 + 1e-12
    return (mu0 - mu1) ** 2 / denom


def run_fisher(
    feature_file: Path,
    class_pairs: list[tuple[str, str]],
    seed: int,
    out_json: Path,
) -> Path:
    data = np.load(feature_file, allow_pickle=True)
    class_names = [str(x) for x in data["class_names"].tolist()]
    labels = data["labels"].astype(np.int64)
    feature_keys = sorted([k for k in data.files if k.startswith("feat_")])
    if not feature_keys:
        raise RuntimeError(f"未在 npz 中找到特征字段: {feature_file}")

    rows = []
    for k in feature_keys:
        x = data[k]
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
            if len(np.unique(y_ab)) < 2 or len(y_ab) < 20:
                continue
            balanced = _balanced_indices(y_ab, seed=seed)
            x_bal = x_ab[balanced]
            y_bal = y_ab[balanced]
            fisher = _fisher_per_channel(x_bal, y_bal, ia, ib)
            topk_idx = np.argsort(fisher)[::-1][:10]
            topk_vals = fisher[topk_idx]
            rows.append(
                {
                    "feature_key": k,
                    "class_pair": f"{a}:{b}",
                    "top10_mean": float(np.mean(topk_vals)),
                    "top10_idx": [int(i) for i in topk_idx.tolist()],
                    "top10_vals": [float(v) for v in topk_vals.tolist()],
                    "n_samples": int(len(y_bal)),
                }
            )

    payload = {
        "time": datetime.now().isoformat(),
        "feature_file": str(feature_file),
        "seed": int(seed),
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = out_json.with_suffix(".csv")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature_key", "class_pair", "top10_mean", "n_samples"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "feature_key": r["feature_key"],
                    "class_pair": r["class_pair"],
                    "top10_mean": r["top10_mean"],
                    "n_samples": r["n_samples"],
                }
            )
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--class-pairs", nargs="+", default=["van:car", "freight_car:truck", "car:bus"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "reports" / "fisher_ratio_results.json"),
    )
    args = parser.parse_args()

    out = run_fisher(
        feature_file=Path(args.features).resolve(),
        class_pairs=_parse_pairs(args.class_pairs),
        seed=int(args.seed),
        out_json=Path(args.output).resolve(),
    )
    print(f"[Diagnostic][Fisher] done: {out}")


if __name__ == "__main__":
    main()
