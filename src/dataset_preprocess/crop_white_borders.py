"""
DroneVehicle 数据集白边裁切工具（启用 mosaic 增强的前置清洗脚本）

设计依据：`数据集白边裁切编码指导.md`

功能概述：
- 对 `data/` 下的 `train/val/test` 三个子集进行离线白边裁切清洗；输出到 `data_croped/`，目录结构与源一致；
- 保证双模态（RGB/IR）严格配对，使用“独立扫描 + 有效区域交集”策略计算裁切窗口；
- 同步转换 OBB 标签（6 列：`class cx cy w h angle`），仅做刚性平移与重新归一化，角度保持不变；
- 生成裁切统计 `data_croped/crop_meta.json` 与异常记录 `data_croped/error.log`；
- 支持多进程加速；支持干运行（不写文件，仅统计与校验）。

使用示例（在项目根目录执行）：
- `python src/dataset_preprocess/crop_white_borders.py --subset all --workers 8`
- `python src/dataset_preprocess/crop_white_borders.py --subset train --threshold 250 --dry-run --limit 100`
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp

import cv2
import numpy as np
from tqdm import tqdm


# ================ 路径与常量约定（动态项目根） ================
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "data"
DEFAULT_DST = ROOT / "data_croped"

# 允许的图片扩展名（按常见格式）
IMG_EXTS = (".jpg", ".jpeg", ".png")


# ===================== 工具函数 =====================
def _to_gray(img: np.ndarray) -> np.ndarray:
    """
    将输入图像转换为灰度图。
    - 对 BGR（3 通道）图像使用 cvtColor；对单通道图像直接返回；对 6 通道（异常）取前 3 通道。
    """
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        c = img.shape[2]
        if c >= 3:
            # 默认 IR 也按 3 通道存储，统一处理
            return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        elif c == 1:
            return img[:, :, 0]
    # 其它情况（如不规整的通道），退化为取第一个通道
    return img[:, :, 0]


def _valid_box(gray: np.ndarray, threshold: int = 250) -> Optional[Tuple[int, int, int, int]]:
    """
    计算灰度图的有效内容包围框：
    - 掩膜定义：像素值 < threshold 视为有效内容（True），否则视为白边（False）；
    - 行/列进行逻辑 OR 投影，定位首/末有效索引；
    返回：(y1, y2, x1, x2)；若无有效内容返回 None。
    """
    if gray is None:
        return None
    mask = gray.astype(np.uint16) < int(threshold)
    if not mask.any():
        return None
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y1 = int(y_indices[0])
    y2 = int(y_indices[-1])
    x1 = int(x_indices[0])
    x2 = int(x_indices[-1])
    return y1, y2, x1, x2


def _intersect_boxes(b_rgb: Tuple[int, int, int, int], b_ir: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
    """
    交集融合策略（更靠内的边界）：
    输入：RGB 与 IR 的有效包围框 (y1,y2,x1,x2)
    输出：裁切窗口 (top, bottom, left, right)
    若交集无效（宽/高<=0）返回 None。
    """
    y1_r, y2_r, x1_r, x2_r = b_rgb
    y1_i, y2_i, x1_i, x2_i = b_ir
    top = max(y1_r, y1_i)
    bottom = min(y2_r, y2_i)
    left = max(x1_r, x1_i)
    right = min(x2_r, x2_i)
    if bottom <= top or right <= left:
        return None
    return top, bottom, left, right


def _list_images(dirp: Path) -> List[Path]:
    """
    列出目录下的所有图像文件路径（按文件名排序）。
    支持扩展名：.jpg/.jpeg/.png
    """
    if not dirp or not dirp.is_dir():
        return []
    xs: List[Path] = []
    for ext in IMG_EXTS:
        xs.extend(dirp.glob(f"*{ext}"))
    return sorted(xs)


def _read_image_candidates(subdir: Path, base: str) -> Optional[Path]:
    """
    在指定子目录下按扩展名候选组合寻找某个基名的实际图像文件路径。
    找到即返回 Path，否则返回 None。
    """
    for ext in IMG_EXTS:
        p = subdir / f"{base}{ext}"
        if p.is_file():
            return p
    return None


def _crop_and_save(img: np.ndarray, rect: Tuple[int, int, int, int], out_path: Path) -> bool:
    """
    根据裁切窗口裁切并保存图像；父目录不存在时自动创建。
    返回写入是否成功。
    """
    top, bottom, left, right = rect
    roi = img[top : bottom + 1, left : right + 1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(out_path.as_posix(), roi))


def _transform_labels(src_label: Optional[Path], out_label: Path, W_old: int, H_old: int, dx: int, dy: int, W_new: int, H_new: int) -> None:
    """
    将 YOLO‑OBB 6 列标签从旧图尺寸变换到新图尺寸：
    - 还原绝对坐标（cx,cy,w,h）；
    - 平移：cx-=dx, cy-=dy；
    - 重新归一化：除以新宽高；
    - 角度保持不变；
    - 若中心点越界（<0 或 >1），则丢弃该行；
    若源标签不存在，则创建空文件。
    """
    out_label.parent.mkdir(parents=True, exist_ok=True)
    if not src_label or not src_label.is_file():
        # 写入空标签文件
        out_label.write_text("", encoding="utf-8")
        return

    lines_out: List[str] = []
    try:
        raw = src_label.read_text(encoding="utf-8")
    except Exception:
        out_label.write_text("", encoding="utf-8")
        return

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            cid = int(parts[0])
            cx_n = float(parts[1])
            cy_n = float(parts[2])
            w_n = float(parts[3])
            h_n = float(parts[4])
            ang = float(parts[5])
        except Exception:
            continue

        # 还原绝对坐标（旧尺寸）
        cx_abs = cx_n * float(W_old)
        cy_abs = cy_n * float(H_old)
        w_abs = w_n * float(W_old)
        h_abs = h_n * float(H_old)

        # 平移（裁切左上角偏移）
        cx_abs2 = cx_abs - float(dx)
        cy_abs2 = cy_abs - float(dy)

        # 重新归一化（新尺寸）
        if W_new <= 0 or H_new <= 0:
            continue
        cx_n2 = cx_abs2 / float(W_new)
        cy_n2 = cy_abs2 / float(H_new)
        w_n2 = w_abs / float(W_new)
        h_n2 = h_abs / float(H_new)

        # 越界丢弃（中心点出界）
        if not (0.0 <= cx_n2 <= 1.0 and 0.0 <= cy_n2 <= 1.0):
            continue

        lines_out.append(f"{cid} {cx_n2:.6f} {cy_n2:.6f} {w_n2:.6f} {h_n2:.6f} {ang:.6f}")

    out_label.write_text("\n".join(lines_out), encoding="utf-8")


def _process_one(args_tuple) -> Tuple[Optional[str], Optional[str]]:
    """
    处理单个样本基名（基于 RGB 目录）：
    输入：参数元组封装所有运行时上下文；输出：(meta_key, rel_id) 或 (None, error_msg)
    - meta_key：如 `x{Left}_y{Top}_w{Wnew}_h{Hnew}`；
    - rel_id：如 `train/0001`；
    - error_msg：发生异常时返回错误消息（包含样本标识）。
    """
    (
        subset,
        base,
        src_rgb_dir,
        src_ir_dir,
        src_lbl_dir,
        dst_rgb_dir,
        dst_ir_dir,
        dst_lbl_dir,
        threshold,
        dry_run,
    ) = args_tuple

    # 解析路径
    rgb_path = _read_image_candidates(src_rgb_dir, base)
    ir_path = _read_image_candidates(src_ir_dir, base)
    if rgb_path is None or ir_path is None:
        return None, f"[{subset}/{base}] 缺失 RGB/IR 图像文件"

    # 读取图像
    img_rgb = cv2.imread(rgb_path.as_posix())
    img_ir = cv2.imread(ir_path.as_posix())
    if img_rgb is None or img_ir is None:
        return None, f"[{subset}/{base}] 图像读取失败"

    H_old, W_old = img_rgb.shape[:2]
    # 灰度与有效框
    g_rgb = _to_gray(img_rgb)
    g_ir = _to_gray(img_ir)
    b_rgb = _valid_box(g_rgb, threshold=threshold)
    b_ir = _valid_box(g_ir, threshold=threshold)
    if b_rgb is None or b_ir is None:
        return None, f"[{subset}/{base}] 有效内容为空（可能全白）"

    inter = _intersect_boxes(b_rgb, b_ir)
    if inter is None:
        return None, f"[{subset}/{base}] 交集裁切窗口无效"

    top, bottom, left, right = inter
    W_new = int(right - left + 1)
    H_new = int(bottom - top + 1)
    if W_new <= 0 or H_new <= 0:
        return None, f"[{subset}/{base}] 裁切尺寸无效"

    # 输出路径
    out_rgb = dst_rgb_dir / f"{base}.jpg"
    out_ir = dst_ir_dir / f"{base}.jpg"
    out_lbl = dst_lbl_dir / f"{base}.txt"

    # 干运行：只返回统计，不写文件
    if dry_run:
        key = f"x{left}_y{top}_w{W_new}_h{H_new}"
        rel_id = f"{subset}/{base}"
        return key, rel_id

    # 执行裁切与保存
    ok1 = _crop_and_save(img_rgb, (top, bottom, left, right), out_rgb)
    ok2 = _crop_and_save(img_ir, (top, bottom, left, right), out_ir)
    if not (ok1 and ok2):
        return None, f"[{subset}/{base}] 图像写入失败"

    # 标签转换（若不存在则创建空标签）
    src_label = (src_lbl_dir / f"{base}.txt") if src_lbl_dir else None
    _transform_labels(src_label, out_lbl, W_old=W_old, H_old=H_old, dx=left, dy=top, W_new=W_new, H_new=H_new)

    key = f"x{left}_y{top}_w{W_new}_h{H_new}"
    rel_id = f"{subset}/{base}"
    return key, rel_id


# ===================== 主流程 =====================
def run_crop(
    src_root: Path = DEFAULT_SRC,
    dst_root: Path = DEFAULT_DST,
    subset: str = "all",
    threshold: int = 250,
    workers: int = max(1, mp.cpu_count() // 2),
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> None:
    """
    主裁切流程：遍历子集、构建目录、并行处理、聚合统计与错误记录。
    参数：
    - src_root：源数据根路径（默认仓库内 `data/`）；
    - dst_root：目标数据根路径（默认 `data_croped/`）；
    - subset：`train`/`val`/`test`/`all`；
    - threshold：白边阈值（默认 250）；
    - workers：并行进程数（默认 CPU/2）；
    - dry_run：干运行开关（默认 False）；
    - limit：每个子集最多处理的样本数（默认 None 全量）。
    """
    subsets = ["train", "val", "test"] if subset.lower() == "all" else [subset.lower()]
    dst_root.mkdir(parents=True, exist_ok=True)
    error_log = dst_root / "error.log"
    meta_json = dst_root / "crop_meta.json"
    all_meta: Dict[str, List[str]] = {}
    errors: List[str] = []

    for ss in subsets:
        # 源目录
        src_rgb_dir = src_root / ss / f"{ss}img"
        src_ir_dir = src_root / ss / f"{ss}imgr"
        src_lbl_dir = src_root / ss / f"{ss}labels_yolo_obb"

        # 目标目录（结构保持一致）
        dst_rgb_dir = dst_root / ss / f"{ss}img"
        dst_ir_dir = dst_root / ss / f"{ss}imgr"
        dst_lbl_dir = dst_root / ss / f"{ss}labels_yolo_obb"
        for d in (dst_rgb_dir, dst_ir_dir, dst_lbl_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 构建基名列表（以 RGB 为准）
        images = _list_images(src_rgb_dir)
        bases = [p.stem for p in images]
        if limit and isinstance(limit, int) and limit > 0:
            bases = bases[:limit]

        # 并行处理参数打包
        jobs = [
            (
                ss,
                b,
                src_rgb_dir,
                src_ir_dir,
                src_lbl_dir,
                dst_rgb_dir,
                dst_ir_dir,
                dst_lbl_dir,
                int(threshold),
                bool(dry_run),
            )
            for b in bases
        ]

        # 执行
        if workers and workers > 1:
            with mp.Pool(processes=int(workers)) as pool:
                for key, val in tqdm(pool.imap_unordered(_process_one, jobs), total=len(jobs), desc=f"{ss}"):
                    if key and val:
                        all_meta.setdefault(key, []).append(val)
                    elif val:
                        errors.append(val)
        else:
            for jt in tqdm(jobs, total=len(jobs), desc=f"{ss}"):
                key, val = _process_one(jt)
                if key and val:
                    all_meta.setdefault(key, []).append(val)
                elif val:
                    errors.append(val)

    # 写出统计与错误日志
    try:
        meta_json.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    if errors:
        try:
            error_log.write_text("\n".join(errors), encoding="utf-8")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="DroneVehicle 数据集白边裁切工具（启用 mosaic 前置清洗）")
    parser.add_argument("--src", type=str, default=str(DEFAULT_SRC), help="源数据根目录，默认 data/")
    parser.add_argument("--dst", type=str, default=str(DEFAULT_DST), help="目标数据根目录，默认 data_croped/")
    parser.add_argument("--subset", type=str, default="all", choices=["train", "val", "test", "all"], help="处理子集")
    parser.add_argument("--threshold", type=int, default=250, help="白边阈值（灰度<阈值视为有效内容）")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2), help="并行进程数")
    parser.add_argument("--dry-run", action="store_true", help="干运行：不写文件，仅统计与校验")
    parser.add_argument("--limit", type=int, default=0, help="每子集最多处理的样本数（0 表示全量）")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    limit = int(args.limit) if int(args.limit) > 0 else None

    run_crop(
        src_root=src_root,
        dst_root=dst_root,
        subset=args.subset,
        threshold=int(args.threshold),
        workers=int(args.workers),
        dry_run=bool(args.dry_run),
        limit=limit,
    )


if __name__ == "__main__":
    main()

