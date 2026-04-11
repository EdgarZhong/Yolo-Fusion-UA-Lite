from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIRS = ["van:car", "freight_car:truck", "car:bus"]
FEATURE_KEY = "feat_fused_p3"


def _require_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise RuntimeError("未检测到 sklearn，请先安装 scikit-learn 后再运行 stratified_analysis.py") from e
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
    out = np.concatenate([sel0, sel1], axis=0)
    rng.shuffle(out)
    return out


def _run_probe(x: np.ndarray, y: np.ndarray, seed: int, folds: int) -> tuple[float | None, float | None, int]:
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None, None, 0

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
        scores.append(float(model.score(x_te[te_bal], y_te[te_bal])))

    n_balanced = int(min(np.sum(y == np.unique(y)[0]), np.sum(y == np.unique(y)[1])) * 2)
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return float(np.mean(scores)), std, n_balanced


def _fisher_per_channel(x: np.ndarray, y: np.ndarray, class0: int, class1: int) -> np.ndarray:
    x0 = x[y == class0]
    x1 = x[y == class1]
    mu0 = np.mean(x0, axis=0)
    mu1 = np.mean(x1, axis=0)
    var0 = np.var(x0, axis=0)
    var1 = np.var(x1, axis=0)
    return (mu0 - mu1) ** 2 / (var0 + var1 + 1e-12)


def _run_fisher(x: np.ndarray, y: np.ndarray, class0: int, class1: int, seed: int) -> tuple[float | None, int]:
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None, 0
    balanced = _balanced_indices(y, seed=seed)
    x_bal = x[balanced]
    y_bal = y[balanced]
    fisher = _fisher_per_channel(x_bal, y_bal, class0, class1)
    topk = np.sort(fisher)[-10:]
    return float(np.mean(topk)), int(len(y_bal))


def _round_or_none(v: float | None, digits: int = 6) -> float | None:
    if v is None:
        return None
    return round(float(v), digits)


def run_stratified_analysis(
    feature_file: Path,
    class_pairs: list[tuple[str, str]],
    brightness_threshold: float,
    seed: int,
    folds: int,
    out_json: Path,
) -> Path:
    data = np.load(feature_file, allow_pickle=True)
    class_names = [str(x) for x in data["class_names"].tolist()]
    labels = data["labels"].astype(np.int64)
    brightness = data["image_brightness"].astype(np.float32)

    if FEATURE_KEY not in data.files:
        raise RuntimeError(f"特征文件中缺少 {FEATURE_KEY}: {feature_file}")

    feat = data[FEATURE_KEY]
    if feat.ndim != 2 or feat.shape[0] != labels.shape[0]:
        raise RuntimeError(f"{FEATURE_KEY} 形状异常: {feat.shape}, labels={labels.shape}")

    bright_mask = brightness >= float(brightness_threshold)
    dark_mask = brightness < float(brightness_threshold)
    condition_masks = {
        "bright": bright_mask,
        "dark": dark_mask,
        "all": np.ones(len(labels), dtype=bool),
    }

    rows: list[dict] = []
    for condition, mask in condition_masks.items():
        feat_sub = feat[mask]
        labels_sub = labels[mask]
        condition_count = int(np.sum(mask))
        for cls_a, cls_b in class_pairs:
            if cls_a not in class_names or cls_b not in class_names:
                continue
            id_a = class_names.index(cls_a)
            id_b = class_names.index(cls_b)
            pair_mask = np.logical_or(labels_sub == id_a, labels_sub == id_b)
            x_ab = feat_sub[pair_mask]
            y_ab = labels_sub[pair_mask]

            probe_mean, probe_std, probe_n = _run_probe(x_ab, y_ab, seed=seed, folds=folds)
            fisher_val, fisher_n = _run_fisher(x_ab, y_ab, class0=id_a, class1=id_b, seed=seed)

            row = {
                "condition": condition,
                "class_pair": f"{cls_a}:{cls_b}",
                "feature_key": FEATURE_KEY,
                "brightness_threshold": float(brightness_threshold),
                "linear_probe_acc": _round_or_none(probe_mean, 4),
                "linear_probe_std": _round_or_none(probe_std, 4),
                "fisher_top10": _round_or_none(fisher_val, 4),
                "n_condition_samples": int(condition_count),
                "n_pair_samples_raw": int(len(y_ab)),
                "n_probe_samples_balanced": int(probe_n),
                "n_fisher_samples_balanced": int(fisher_n),
                "n_total_samples": int(len(labels)),
                "n_bright_samples": int(np.sum(bright_mask)),
                "n_dark_samples": int(np.sum(dark_mask)),
            }
            rows.append(row)
            print(
                f"[Stratified] {condition} | {cls_a}:{cls_b} | "
                f"probe={row['linear_probe_acc']}±{row['linear_probe_std']} | "
                f"fisher={row['fisher_top10']}"
            )

    payload = {
        "time": datetime.now().isoformat(),
        "feature_file": str(feature_file),
        "seed": int(seed),
        "folds": int(folds),
        "feature_key": FEATURE_KEY,
        "brightness_threshold": float(brightness_threshold),
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = out_json.with_suffix(".csv")
    fieldnames = [
        "condition",
        "class_pair",
        "feature_key",
        "brightness_threshold",
        "linear_probe_acc",
        "linear_probe_std",
        "fisher_top10",
        "n_condition_samples",
        "n_pair_samples_raw",
        "n_probe_samples_balanced",
        "n_fisher_samples_balanced",
        "n_total_samples",
        "n_bright_samples",
        "n_dark_samples",
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
    parser.add_argument("--brightness-threshold", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "reports" / "stratified_analysis_results.json"),
    )
    args = parser.parse_args()

    out = run_stratified_analysis(
        feature_file=Path(args.features).resolve(),
        class_pairs=_parse_pairs(list(args.class_pairs)),
        brightness_threshold=float(args.brightness_threshold),
        seed=int(args.seed),
        folds=int(args.folds),
        out_json=Path(args.output).resolve(),
    )
    print(f"[Diagnostic][Stratified] done: {out}")


if __name__ == "__main__":
    main()
