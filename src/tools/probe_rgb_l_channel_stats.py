from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


IMG_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}
ROOT = Path(__file__).resolve().parents[2]


def collect_images(img_dir: Path) -> list[Path]:
    return sorted(p for p in img_dir.rglob("*.*") if p.suffix.lower() in IMG_EXTS)


def compute_l_mean(image_path: Path) -> float:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"无法读取图像: {image_path}")
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    return float(np.mean(l_channel))


def write_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = ["image", "l_mean"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(values: np.ndarray, summary_path: Path, threshold: float) -> dict:
    if values.size == 0:
        raise RuntimeError("未得到任何 L 通道均值数据")
    low_count = int(np.sum(values < threshold))
    summary = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "threshold": float(threshold),
        "low_count": low_count,
        "low_ratio": float(low_count / values.size),
    }
    lines = [
        f"count,{summary['count']}",
        f"mean,{summary['mean']:.6f}",
        f"median,{summary['median']:.6f}",
        f"threshold,{summary['threshold']:.2f}",
        f"low_count,{summary['low_count']}",
        f"low_ratio,{summary['low_ratio']:.6f}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def plot_hist(values: np.ndarray, out_path: Path, threshold: float) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, color="#4e79a7", alpha=0.85, edgecolor="white")
    plt.axvline(threshold, color="#e15759", linestyle="--", linewidth=1.5, label=f"L<{threshold:.0f} 阈值")
    plt.xlabel("L 通道均值")
    plt.ylabel("图像数量")
    plt.title("testimg RGB 图像 LAB-L 均值分布")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, default=str(ROOT / "data_croped" / "test" / "testimg"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "result" / "dataset_probe" / "rgb_l_channel"))
    parser.add_argument("--threshold", type=float, default=40.0)
    args = parser.parse_args()

    img_dir = Path(args.img_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(img_dir)
    if not images:
        raise RuntimeError(f"未在目录中找到图像: {img_dir}")

    rows: list[dict] = []
    values: list[float] = []
    for image_path in images:
        l_mean = compute_l_mean(image_path)
        rows.append({"image": str(image_path), "l_mean": f"{l_mean:.6f}"})
        values.append(l_mean)

    arr = np.asarray(values, dtype=np.float64)
    csv_path = out_dir / "rgb_l_mean_per_image.csv"
    hist_path = out_dir / "rgb_l_mean_hist.png"
    summary_path = out_dir / "rgb_l_summary.txt"

    write_csv(rows, csv_path)
    summary = write_summary(arr, summary_path, args.threshold)
    plot_hist(arr, hist_path, args.threshold)

    print(f"[RGB-L] 图像数: {summary['count']}")
    print(f"[RGB-L] 均值: {summary['mean']:.6f}")
    print(f"[RGB-L] 中位数: {summary['median']:.6f}")
    print(f"[RGB-L] L<{args.threshold:.0f} 数量: {summary['low_count']}")
    print(f"[RGB-L] L<{args.threshold:.0f} 占比: {summary['low_ratio']:.6f}")
    print(f"[RGB-L] CSV: {csv_path}")
    print(f"[RGB-L] 直方图: {hist_path}")
    print(f"[RGB-L] 摘要: {summary_path}")


if __name__ == "__main__":
    main()
