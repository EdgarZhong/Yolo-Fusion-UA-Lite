from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def _require_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise RuntimeError("未检测到 sklearn，请先安装 scikit-learn 后再运行 linear_probe.py") from e
    return LogisticRegression, StratifiedKFold, make_pipeline, StandardScaler


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


def _run_probe(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    folds: int,
) -> tuple[float, float]:
    LogisticRegression, StratifiedKFold, make_pipeline, StandardScaler = _require_sklearn()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for fi, (tr, te) in enumerate(skf.split(x, y)):
        x_tr, y_tr = x[tr], y[tr]
        x_te, y_te = x[te], y[te]
        tr_bal = _balanced_indices(y_tr, seed + fi * 37 + 1)
        te_bal = _balanced_indices(y_te, seed + fi * 37 + 2)
        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=3000, random_state=seed + fi, solver="lbfgs"),
        )
        model.fit(x_tr[tr_bal], y_tr[tr_bal])
        acc = float(model.score(x_te[te_bal], y_te[te_bal]))
        scores.append(acc)
    return float(np.mean(scores)), float(np.std(scores, ddof=1) if len(scores) > 1 else 0.0)


def run_linear_probe(
    feature_file: Path,
    class_pairs: list[tuple[str, str]],
    seed: int,
    folds: int,
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
            mean_acc, std_acc = _run_probe(x_ab, y_ab, seed=seed, folds=folds)
            rows.append(
                {
                    "feature_key": k,
                    "class_pair": f"{a}:{b}",
                    "acc_mean": mean_acc,
                    "acc_std": std_acc,
                    "n_samples": int(len(y_ab)),
                }
            )

    payload = {
        "time": datetime.now().isoformat(),
        "feature_file": str(feature_file),
        "seed": int(seed),
        "folds": int(folds),
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = out_json.with_suffix(".csv")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature_key", "class_pair", "acc_mean", "acc_std", "n_samples"])
        w.writeheader()
        w.writerows(rows)
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--class-pairs", nargs="+", default=["van:car", "freight_car:truck", "car:bus"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "reports" / "linear_probe_results.json"),
    )
    args = parser.parse_args()

    out = run_linear_probe(
        feature_file=Path(args.features).resolve(),
        class_pairs=_parse_pairs(args.class_pairs),
        seed=int(args.seed),
        folds=int(args.folds),
        out_json=Path(args.output).resolve(),
    )
    print(f"[Diagnostic][LinearProbe] done: {out}")


ROOT = Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    main()
