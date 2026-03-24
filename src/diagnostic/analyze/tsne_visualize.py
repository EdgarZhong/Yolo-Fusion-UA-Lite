from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def _require_sklearn_tsne():
    try:
        from sklearn.manifold import TSNE
    except Exception as e:
        raise RuntimeError("未检测到 sklearn，请先安装 scikit-learn 后再运行 tsne_visualize.py") from e
    return TSNE


def _parse_pairs(pairs: list[str]) -> list[tuple[str, str]]:
    out = []
    for p in pairs:
        if ":" not in p:
            raise ValueError(f"class pair 格式错误: {p}")
        a, b = p.split(":", 1)
        out.append((a.strip(), b.strip()))
    return out


def _balanced_indices(y: np.ndarray, seed: int, max_per_class: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uniq = np.unique(y)
    if len(uniq) != 2:
        raise ValueError("当前实现仅支持二分类 pair")
    idx0 = np.where(y == uniq[0])[0]
    idx1 = np.where(y == uniq[1])[0]
    n = min(len(idx0), len(idx1), max_per_class)
    sel0 = rng.choice(idx0, size=n, replace=False)
    sel1 = rng.choice(idx1, size=n, replace=False)
    out = np.concatenate([sel0, sel1], axis=0)
    rng.shuffle(out)
    return out


def run_tsne(
    feature_file: Path,
    feature_key: str,
    class_pairs: list[tuple[str, str]],
    perplexities: list[int],
    seed: int,
    max_per_class: int,
    out_dir: Path,
) -> Path:
    TSNE = _require_sklearn_tsne()
    data = np.load(feature_file, allow_pickle=True)
    class_names = [str(x) for x in data["class_names"].tolist()]
    labels = data["labels"].astype(np.int64)
    if feature_key not in data.files:
        raise RuntimeError(f"特征键不存在: {feature_key}")
    x_all = data[feature_key]
    if x_all.ndim != 2 or x_all.shape[0] != labels.shape[0]:
        raise RuntimeError(f"特征形状不合法: {feature_key}, shape={x_all.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for a, b in class_pairs:
        if a not in class_names or b not in class_names:
            continue
        ia = class_names.index(a)
        ib = class_names.index(b)
        mask = np.logical_or(labels == ia, labels == ib)
        x_ab = x_all[mask]
        y_ab = labels[mask]
        if len(np.unique(y_ab)) < 2 or len(y_ab) < 20:
            continue

        idx = _balanced_indices(y_ab, seed=seed, max_per_class=max_per_class)
        x_bal = x_ab[idx]
        y_bal = y_ab[idx]
        y_str = np.array([a if yi == ia else b for yi in y_bal], dtype=object)

        for perp in perplexities:
            eff_perp = min(float(perp), max(5.0, float(len(y_bal) - 1)))
            if eff_perp >= float(len(y_bal)):
                continue
            tsne = TSNE(
                n_components=2,
                perplexity=eff_perp,
                random_state=seed,
                init="pca",
                learning_rate="auto",
            )
            z = tsne.fit_transform(x_bal)

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            for cname, color in [(a, "#e15759"), (b, "#4e79a7")]:
                cmask = y_str == cname
                ax.scatter(z[cmask, 0], z[cmask, 1], s=8, alpha=0.55, label=cname, c=color)
            ax.set_title(f"t-SNE {feature_key} | {a} vs {b} | perplexity={eff_perp:.1f}")
            ax.set_xlabel("dim-1")
            ax.set_ylabel("dim-2")
            ax.legend()
            fig.tight_layout()

            png = out_dir / f"tsne_{feature_key}_{a}_vs_{b}_p{perp}.png"
            fig.savefig(png, dpi=180)
            plt.close(fig)
            records.append(
                {
                    "feature_key": feature_key,
                    "class_pair": f"{a}:{b}",
                    "perplexity": float(eff_perp),
                    "n_samples": int(len(y_bal)),
                    "file": str(png),
                }
            )

    report = {
        "time": datetime.now().isoformat(),
        "feature_file": str(feature_file),
        "feature_key": feature_key,
        "seed": int(seed),
        "records": records,
    }
    out_json = out_dir / f"tsne_{feature_key}_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--feature-key", type=str, default="feat_fused_p3")
    parser.add_argument("--class-pairs", nargs="+", default=["van:car"])
    parser.add_argument("--perplexities", nargs="+", type=int, default=[30, 15, 50])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=2000)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "figures"),
    )
    args = parser.parse_args()

    out = run_tsne(
        feature_file=Path(args.features).resolve(),
        feature_key=str(args.feature_key),
        class_pairs=_parse_pairs(args.class_pairs),
        perplexities=[int(v) for v in args.perplexities],
        seed=int(args.seed),
        max_per_class=int(args.max_per_class),
        out_dir=Path(args.out_dir).resolve(),
    )
    print(f"[Diagnostic][TSNE] done: {out}")


if __name__ == "__main__":
    main()
