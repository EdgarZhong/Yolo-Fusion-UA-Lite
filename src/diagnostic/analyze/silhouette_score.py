from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def _require_sklearn():
    try:
        from sklearn.metrics import silhouette_score
    except Exception as e:
        raise RuntimeError("未检测到 sklearn，请先安装 scikit-learn 后再运行 silhouette_score.py") from e
    return silhouette_score


def _parse_pairs(pairs: list[str]) -> list[tuple[str, str]]:
    out = []
    for p in pairs:
        if ":" not in p:
            raise ValueError(f"class pair 格式错误: {p}")
        a, b = p.split(":", 1)
        out.append((a.strip(), b.strip()))
    return out


def _balanced_indices(y: np.ndarray, seed: int, max_total_samples: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uniq = np.unique(y)
    if len(uniq) != 2:
        raise ValueError("当前实现仅支持二分类 pair")
    idx0 = np.where(y == uniq[0])[0]
    idx1 = np.where(y == uniq[1])[0]
    n = min(len(idx0), len(idx1), max_total_samples // 2)
    sel0 = rng.choice(idx0, size=n, replace=False)
    sel1 = rng.choice(idx1, size=n, replace=False)
    out = np.concatenate([sel0, sel1], axis=0)
    rng.shuffle(out)
    return out


def run_silhouette(
    feature_file: Path,
    class_pairs: list[tuple[str, str]],
    seed: int,
    max_total_samples: int,
    feature_keys: list[str],
    out_json: Path,
) -> Path:
    silhouette_score = _require_sklearn()
    data = np.load(feature_file, allow_pickle=True)
    class_names = [str(x) for x in data["class_names"].tolist()]
    labels = data["labels"].astype(np.int64)

    rows = []
    for feature_key in feature_keys:
        if feature_key not in data.files:
            continue
        x = data[feature_key]
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
            idx = _balanced_indices(y_ab, seed=seed, max_total_samples=max_total_samples)
            x_bal = x_ab[idx]
            y_bal = y_ab[idx]
            score = float(silhouette_score(x_bal, y_bal))
            rows.append(
                {
                    "feature_key": feature_key,
                    "class_pair": f"{a}:{b}",
                    "silhouette": score,
                    "n_samples": int(len(y_bal)),
                }
            )

    payload = {
        "time": datetime.now().isoformat(),
        "feature_file": str(feature_file),
        "seed": int(seed),
        "max_total_samples": int(max_total_samples),
        "feature_keys": feature_keys,
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = out_json.with_suffix(".csv")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature_key", "class_pair", "silhouette", "n_samples"])
        w.writeheader()
        w.writerows(rows)
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--class-pairs", nargs="+", default=["van:car", "freight_car:truck", "car:bus"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-total-samples", type=int, default=5000)
    parser.add_argument("--feature-keys", nargs="+", default=["feat_fused_p3"])
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "reports" / "silhouette_score_results.json"),
    )
    args = parser.parse_args()

    out = run_silhouette(
        feature_file=Path(args.features).resolve(),
        class_pairs=_parse_pairs(args.class_pairs),
        seed=int(args.seed),
        max_total_samples=int(args.max_total_samples),
        feature_keys=[str(v) for v in args.feature_keys],
        out_json=Path(args.output).resolve(),
    )
    print(f"[Diagnostic][Silhouette] done: {out}")


if __name__ == "__main__":
    main()
