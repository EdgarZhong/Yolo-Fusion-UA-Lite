from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import ttest_ind


ROOT = Path(__file__).resolve().parents[2]
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".dng", ".mpo", ".pfm"]


def load_names(dataset_yaml: Path) -> list[str]:
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not isinstance(names, list):
        raise RuntimeError(f"数据集 names 解析失败: {dataset_yaml}")
    return [str(x) for x in names]


def resolve_test_dirs(dataset_yaml: Path) -> tuple[Path, Path, Path]:
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", ROOT))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    test_img = Path(data["test"])
    test_img = (root / test_img).resolve() if not test_img.is_absolute() else test_img.resolve()
    test_ir = test_img.parent / f"{test_img.name}r"
    subset = test_img.parent.name
    label_dir = test_img.parent / f"{subset}labels_yolo_obb"
    return test_img, test_ir, label_dir


def find_image_file(dir_path: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = dir_path / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def obb_to_polygon(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    dx = w / 2.0
    dy = h / 2.0
    corners = np.array(
        [
            [-dx, -dy],
            [dx, -dy],
            [dx, dy],
            [-dx, dy],
        ],
        dtype=np.float32,
    )
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    pts = corners @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts


def parse_label_file(
    label_file: Path,
    img_w: int,
    img_h: int,
    car_id: int,
    van_id: int,
) -> list[tuple[int, np.ndarray]]:
    targets: list[tuple[int, np.ndarray]] = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 6:
            continue
        cls_id = int(float(parts[0]))
        if cls_id not in (car_id, van_id):
            continue
        cx = float(parts[1]) * img_w
        cy = float(parts[2]) * img_h
        bw = float(parts[3]) * img_w
        bh = float(parts[4]) * img_h
        ang = float(parts[5])
        poly = obb_to_polygon(cx, cy, bw, bh, ang)
        targets.append((cls_id, poly))
    return targets


def region_stats(ir_gray: np.ndarray, poly: np.ndarray) -> tuple[float, float, int] | None:
    h, w = ir_gray.shape[:2]
    pts = np.round(poly).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=255)
    pixels = ir_gray[mask > 0]
    if pixels.size == 0:
        return None
    return float(np.mean(pixels)), float(np.var(pixels)), int(pixels.size)


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["image", "label_file", "class_name", "pixel_mean", "pixel_var", "pixel_count"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_distributions(
    car_means: np.ndarray,
    van_means: np.ndarray,
    car_vars: np.ndarray,
    van_vars: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].hist(car_means, bins=50, alpha=0.6, label="car", color="#4e79a7")
    axes[0].hist(van_means, bins=50, alpha=0.6, label="van", color="#e15759")
    axes[0].set_title("IR 区域像素均值分布")
    axes[0].set_xlabel("pixel mean")
    axes[0].set_ylabel("count")
    axes[0].legend()

    axes[1].hist(car_vars, bins=50, alpha=0.6, label="car", color="#4e79a7")
    axes[1].hist(van_vars, bins=50, alpha=0.6, label="van", color="#e15759")
    axes[1].set_title("IR 区域像素方差分布")
    axes[1].set_xlabel("pixel variance")
    axes[1].set_ylabel("count")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summarize(
    car_means: np.ndarray,
    van_means: np.ndarray,
    car_vars: np.ndarray,
    van_vars: np.ndarray,
    out_path: Path,
) -> None:
    mean_p = float("nan")
    var_p = float("nan")
    if car_means.size >= 2 and van_means.size >= 2:
        mean_p = float(ttest_ind(car_means, van_means, equal_var=False).pvalue)
    if car_vars.size >= 2 and van_vars.size >= 2:
        var_p = float(ttest_ind(car_vars, van_vars, equal_var=False).pvalue)
    lines = [
        f"car_count,{car_means.size}",
        f"van_count,{van_means.size}",
        f"car_mean_avg,{float(np.mean(car_means)):.6f}" if car_means.size else "car_mean_avg,nan",
        f"van_mean_avg,{float(np.mean(van_means)):.6f}" if van_means.size else "van_mean_avg,nan",
        f"car_var_avg,{float(np.mean(car_vars)):.6f}" if car_vars.size else "car_var_avg,nan",
        f"van_var_avg,{float(np.mean(van_vars)):.6f}" if van_vars.size else "van_var_avg,nan",
        f"ttest_p_mean,{'nan' if math.isnan(mean_p) else f'{mean_p:.12g}'}",
        f"ttest_p_var,{'nan' if math.isnan(var_p) else f'{var_p:.12g}'}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-yaml",
        type=str,
        default=str(ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "result" / "dataset_probe" / "ir_van_car_regions"),
    )
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    names = load_names(data_yaml)
    if "car" not in names or "van" not in names:
        raise RuntimeError(f"类别名中未找到 car/van: {names}")
    car_id = names.index("car")
    van_id = names.index("van")

    _, ir_dir, label_dir = resolve_test_dirs(data_yaml)
    if not ir_dir.exists():
        raise FileNotFoundError(f"IR 目录不存在: {ir_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {label_dir}")

    rows: list[dict] = []
    car_means: list[float] = []
    van_means: list[float] = []
    car_vars: list[float] = []
    van_vars: list[float] = []

    label_files = sorted(label_dir.rglob("*.txt"))
    if not label_files:
        raise RuntimeError(f"未找到标签文件: {label_dir}")

    for label_file in label_files:
        stem = label_file.stem
        ir_path = find_image_file(ir_dir, stem)
        if ir_path is None:
            continue
        ir_img = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        if ir_img is None:
            continue
        h, w = ir_img.shape[:2]
        targets = parse_label_file(label_file, w, h, car_id, van_id)
        for cls_id, poly in targets:
            stats = region_stats(ir_img, poly)
            if stats is None:
                continue
            m, v, count = stats
            class_name = "car" if cls_id == car_id else "van"
            rows.append(
                {
                    "image": str(ir_path),
                    "label_file": str(label_file),
                    "class_name": class_name,
                    "pixel_mean": f"{m:.6f}",
                    "pixel_var": f"{v:.6f}",
                    "pixel_count": str(count),
                }
            )
            if cls_id == car_id:
                car_means.append(m)
                car_vars.append(v)
            else:
                van_means.append(m)
                van_vars.append(v)

    if not rows:
        raise RuntimeError("未统计到任何 car/van IR 区域样本")

    car_means_arr = np.asarray(car_means, dtype=np.float64)
    van_means_arr = np.asarray(van_means, dtype=np.float64)
    car_vars_arr = np.asarray(car_vars, dtype=np.float64)
    van_vars_arr = np.asarray(van_vars, dtype=np.float64)

    csv_path = out_dir / "ir_van_car_region_stats.csv"
    fig_path = out_dir / "ir_van_car_distribution.png"
    summary_path = out_dir / "ir_van_car_ttest_summary.txt"

    write_csv(rows, csv_path)
    plot_distributions(car_means_arr, van_means_arr, car_vars_arr, van_vars_arr, fig_path)
    summarize(car_means_arr, van_means_arr, car_vars_arr, van_vars_arr, summary_path)

    print(f"[IR-Region] car 实例数: {car_means_arr.size}")
    print(f"[IR-Region] van 实例数: {van_means_arr.size}")
    print(f"[IR-Region] CSV: {csv_path}")
    print(f"[IR-Region] 分布图: {fig_path}")
    print(f"[IR-Region] 检验摘要: {summary_path}")


if __name__ == "__main__":
    main()
