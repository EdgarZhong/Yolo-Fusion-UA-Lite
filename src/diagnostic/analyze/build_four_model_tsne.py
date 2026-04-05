from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CLASS_MAP = {"car": 0, "truck": 1, "bus": 2, "van": 3, "freight_car": 4}
MODEL_SPECS = [
    ("Exp-0", "exp0_se.npz", "Exp-0 (SE)"),
    ("Exp-A", "expa_no_attn.npz", "Exp-A (None)"),
    ("Exp-B", "expb_coordattn.npz", "Exp-B (CoordAttn)"),
    ("Exp-C", "expc_simam.npz", "Exp-C (SimAM)"),
]


def _require_sklearn():
    try:
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score
    except Exception as e:
        raise RuntimeError("未检测到 sklearn，请先安装 scikit-learn 后再运行 build_four_model_tsne.py") from e
    return TSNE, silhouette_score


def _sample_pair(feat: np.ndarray, labels: np.ndarray, pair: str, seed: int, total_limit: int) -> tuple[np.ndarray, np.ndarray]:
    cls_a, cls_b = pair.split(":")
    id_a, id_b = CLASS_MAP[cls_a], CLASS_MAP[cls_b]
    mask = np.isin(labels, [id_a, id_b])
    feat_sub = feat[mask]
    labels_sub = labels[mask]
    rng = np.random.default_rng(seed)
    n_per_class = min(np.sum(labels_sub == id_a), np.sum(labels_sub == id_b), total_limit // 2)
    idx_a = rng.choice(np.where(labels_sub == id_a)[0], size=n_per_class, replace=False)
    idx_b = rng.choice(np.where(labels_sub == id_b)[0], size=n_per_class, replace=False)
    idx = np.concatenate([idx_a, idx_b], axis=0)
    rng.shuffle(idx)
    return feat_sub[idx], labels_sub[idx]


def _embedding_metrics(emb: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    uniq = np.unique(labels)
    centers = []
    intra = []
    for u in uniq:
        cur = emb[labels == u]
        center = cur.mean(axis=0)
        centers.append(center)
        intra.append(float(np.mean(np.linalg.norm(cur - center, axis=1))))
    center_dist = float(np.linalg.norm(centers[0] - centers[1])) if len(centers) == 2 else 0.0
    return {
        "center_distance": center_dist,
        "intra_mean": float(np.mean(intra)) if intra else 0.0,
        "sep_ratio": center_dist / max(1e-12, float(np.mean(intra)) if intra else 1.0),
    }


def build_tsne(
    features_dir: Path,
    figures_dir: Path,
    pairs: list[str],
    perplexities: list[int],
    seed: int,
    total_limit: int,
) -> Path:
    TSNE, silhouette_score = _require_sklearn()
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_records = []

    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for model_key, file_name, _ in MODEL_SPECS:
        data = np.load(features_dir / file_name, allow_pickle=True)
        feat = data["feat_fused_p3"]
        labels = data["labels"]
        for pair in pairs:
            cache[(model_key, pair)] = _sample_pair(feat, labels, pair=pair, seed=seed, total_limit=total_limit)

    for pair in pairs:
        cls_a, cls_b = pair.split(":")
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"t-SNE: {cls_a} vs {cls_b} (feat_fused_p3, perplexity=30)", fontsize=14)

        for ax, (model_key, _, display_name) in zip(axes.flat, MODEL_SPECS):
            feat_sub, labels_sub = cache[(model_key, pair)]
            emb_p30 = None
            p30_score = None
            p30_sep = None
            for perp in perplexities:
                tsne = TSNE(n_components=2, perplexity=float(perp), random_state=seed, init="pca", learning_rate="auto")
                emb = tsne.fit_transform(feat_sub)
                sil = float(silhouette_score(emb, labels_sub))
                metric = _embedding_metrics(emb, labels_sub)
                summary_records.append(
                    {
                        "model": model_key,
                        "pair": pair,
                        "perplexity": int(perp),
                        "n_samples": int(len(labels_sub)),
                        "silhouette_2d": sil,
                        "center_distance": metric["center_distance"],
                        "intra_mean": metric["intra_mean"],
                        "sep_ratio": metric["sep_ratio"],
                    }
                )
                if int(perp) == 30:
                    emb_p30 = emb
                    p30_score = sil
                    p30_sep = metric["sep_ratio"]

            id_a, id_b = CLASS_MAP[cls_a], CLASS_MAP[cls_b]
            colors = {id_a: "#1f77b4", id_b: "#d62728"}
            for cid, cname in [(id_a, cls_a), (id_b, cls_b)]:
                cmask = labels_sub == cid
                ax.scatter(emb_p30[cmask, 0], emb_p30[cmask, 1], c=colors[cid], s=3, alpha=0.4, label=cname)
            ax.set_title(f"{display_name}\n2D-sil={p30_score:.3f}, sep={p30_sep:.3f}", fontsize=12)
            ax.legend(fontsize=10, markerscale=3)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        out_png = figures_dir / ("four_model_tsne_van_car.png" if pair == "van:car" else "four_model_tsne_fc_truck.png")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)

    summary_path = figures_dir / "four_model_tsne_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "time": datetime.now().isoformat(),
                "pairs": pairs,
                "perplexities": perplexities,
                "seed": int(seed),
                "total_limit": int(total_limit),
                "rows": summary_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-dir",
        type=str,
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "features"),
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(ROOT / "src" / "diagnostic" / "outputs" / "figures"),
    )
    parser.add_argument("--pairs", nargs="+", default=["van:car", "freight_car:truck"])
    parser.add_argument("--perplexities", nargs="+", type=int, default=[15, 30, 50])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-limit", type=int, default=5000)
    args = parser.parse_args()

    out = build_tsne(
        features_dir=Path(args.features_dir).resolve(),
        figures_dir=Path(args.figures_dir).resolve(),
        pairs=[str(v) for v in args.pairs],
        perplexities=[int(v) for v in args.perplexities],
        seed=int(args.seed),
        total_limit=int(args.total_limit),
    )
    print(f"[Diagnostic][TSNE-Compare] done: {out}")


if __name__ == "__main__":
    main()
