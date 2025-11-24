"""
通用推理可视化脚本：从 `result/<run_name>/predictions.json` 加载预测结果，
并从 `data/test` 加载左右配对的 RGB/IR 原始图，在窗口中左右并排显示，
将推理结果的旋转框（poly 多边形）以绿色绘制在两张图片上，通过左右方向键切换样本。

使用示例：
- `python src/testing/view_inference.py --pred-dir result/formal-baseline`

参数说明：
- `--pred-dir` 指定包含 `predictions.json` 的结果目录（必需），例如 `result/formal-3`
- `--data-root` 指定数据根目录（默认 `data/test`），脚本会在其中自动查找 `testimg/` 与 `testimgr/`
- `--start-index` 指定起始样本下标（默认 0）
- `--conf` 指定绘制的置信度过滤阈值（默认 0.5，仅绘制得分不低于此值的框）

绘制与交互：
- 绿色多边形为推理结果；左为 RGB，右为 IR；窗口标题显示当前样本信息
- 按左/右方向键切换样本；按 `q` 或 `ESC` 退出
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _find_img_dirs(sub_root: Path):
    """
    在指定的子集根目录下查找包含图像的子目录。
    约定：目录名中包含 "img" 的均视为图像目录，优先使用 `testimg/` 与 `testimgr/`。
    返回：(rgb_img_dir, ir_img_dir)
    """
    # 1) 首选严格目录名（与 README 保持一致）：testimg / testimgr
    rgb_pref = sub_root / "testimg"
    ir_pref = sub_root / "testimgr"
    if rgb_pref.is_dir() and ir_pref.is_dir():
        return rgb_pref, ir_pref

    # 2) 其次按包含规则查找（容错）：名称含 "img" 的目录
    img_dirs = []
    for p in sub_root.iterdir():
        if p.is_dir() and "img" in p.name.lower():
            img_dirs.append(p)
    img_dirs = sorted(img_dirs)
    if not img_dirs:
        return None, None
    # 优先匹配带 r 的红外目录
    rgb_dir = None
    ir_dir = None
    for p in img_dirs:
        ln = p.name.lower()
        if ln.endswith("imgr") or ln.endswith("img_r"):
            ir_dir = p
        elif ln.endswith("img"):
            rgb_dir = p
    if rgb_dir is None:
        rgb_dir = img_dirs[0]
    if ir_dir is None:
        # 若未找到 ir 目录，则尝试第二个，否则回退为 rgb 目录（可能仅有单路）
        ir_dir = img_dirs[1] if len(img_dirs) > 1 else rgb_dir
    return rgb_dir, ir_dir


def _list_images(dirp: Path):
    """
    列出指定目录下的所有图像文件，按文件名排序。
    支持扩展名：.jpg/.jpeg/.png
    """
    if dirp is None or not dirp.is_dir():
        return []
    xs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        xs.extend(dirp.glob(ext))
    return sorted(xs)


def _ensure_image(path: Path | None) -> np.ndarray:
    """
    读取图像；若路径不存在或读取失败，返回 512x512 的黑底占位图。
    """
    img = cv2.imread(path.as_posix()) if path and path.is_file() else None
    if img is None:
        img = np.zeros((512, 512, 3), dtype=np.uint8)
    return img


def _draw_polys(img: np.ndarray, polys: np.ndarray, labels: list[str] | None = None, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制一组多边形，并在框附近绘制小字类名。

    参数：
    - img：输入 BGR 图
    - polys：形如 (N,8) 的坐标数组
    - labels：长度为 N 的字符串列表；若为 None 或长度不一致则不绘制文字
    - color/thickness：绘制颜色与线宽
    """
    out = img.copy()
    for idx, pts in enumerate(polys):
        pts2 = pts.reshape(-1, 2).astype(np.int32)
        cv2.polylines(out, [pts2], True, color, thickness)
        if labels and idx < len(labels):
            # 文本位置取多边形质心，略微上移避免遮挡
            cx = int(np.mean(pts2[:, 0]))
            cy = int(np.mean(pts2[:, 1]))
            pos = (max(0, cx - 10), max(0, cy - 10))
            cv2.putText(out, labels[idx], pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def _resize_pair_left_right(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    将左右两张图片按高度对齐后并排拼接输出。
    """
    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    H = max(h1, h2)

    def _scale(img, H):
        h, w = img.shape[:2]
        if h == H:
            return img
        r = H / float(h)
        return cv2.resize(img, (int(w * r), H), interpolation=cv2.INTER_LINEAR)

    l2 = _scale(left, H)
    r2 = _scale(right, H)
    return np.concatenate([l2, r2], axis=1)


def _group_predictions(pred_json: Path):
    """
    读取 `predictions.json` 并按 `image_id` 分组，返回：
    - `groups`: dict[str, list[dict]]，其中每个字典元素包含 `poly` 等字段
    - `keys`: list[str]，图像键的有序列表（按出现顺序）
    说明：
    - `image_id` 可能为文件名 stem 或复合字符串（如 'rgb|ir'）；本函数保持原样分组。
    - 绘制阶段再根据键值解析对应的 RGB/IR 图像路径。
    """
    if not pred_json.is_file():
        raise FileNotFoundError(f"未找到 predictions.json：{pred_json.as_posix()}")
    data = json.loads(Path(pred_json).read_text(encoding="utf-8"))
    groups = {}
    keys = []
    for d in data:
        k = str(d.get("image_id", ""))
        if k not in groups:
            groups[k] = []
            keys.append(k)
        groups[k].append(d)
    return groups, keys


def _read_classes_map(data_root: Path):
    """
    读取类别映射表：优先从 `data_root/classes.txt`，若不存在再尝试 `data_root.parent/classes.txt`。
    文件格式：每行 `<id> <name>`，例如：`0 car`。
    返回：`dict[int, str]`，不存在时返回空字典。
    """
    candidates = [data_root / "classes.txt", data_root.parent / "classes.txt"]
    id2name = {}
    for p in candidates:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        try:
                            cid = int(parts[0])
                        except Exception:
                            continue
                        id2name[cid] = parts[1]
                break
            except Exception:
                pass
    return id2name

def _build_pair_index(rgb_dir: Path, ir_dir: Path):
    """
    基于 `rgb_dir` 与 `ir_dir` 的文件列表构建样本对索引：
    返回：`base_to_pair`: dict[str, tuple[Path|None, Path|None]]，键为不含扩展名的文件基名。
    """
    xs_rgb = _list_images(rgb_dir)
    xs_ir = _list_images(ir_dir)
    base_to_pair = {}
    int_to_base = {}
    for p in xs_rgb:
        base = p.stem
        base_to_pair.setdefault(base, [None, None])[0] = p
        # 若基名为纯数字（可能含前导 0），记录其整数映射，便于 predictions.json 的整型 image_id 匹配
        if base.isdigit():
            try:
                int_to_base[int(base)] = base
            except Exception:
                pass
    for p in xs_ir:
        base = p.stem
        base_to_pair.setdefault(base, [None, None])[1] = p
        if base.isdigit():
            try:
                int_val = int(base)
                # 以 RGB 的记录为主，若此前未记录则补充
                int_to_base.setdefault(int_val, base)
            except Exception:
                pass
    # 转为 tuple
    for k in list(base_to_pair.keys()):
        rgb_p, ir_p = base_to_pair[k]
        base_to_pair[k] = (rgb_p, ir_p)
    return base_to_pair, int_to_base


def _resolve_pair_for_key(key: str, base_to_pair: dict, int_to_base: dict):
    """
    根据 `predictions.json` 的 `image_id` 解析对应的 (rgb_path, ir_path)。
    规则：
    - 若包含管道符 '|': 视为复合路径，取两侧的文件名 stem 做候选；
    - 否则：直接以 `key` 作为 stem 匹配；若失败则尝试仅取文件名部分作为 stem。
    """
    candidates = []
    if "|" in key:
        left, right = key.split("|", 1)
        left_base = Path(left).stem
        right_base = Path(right).stem
        candidates.extend([left_base, right_base])
    else:
        candidates.append(Path(key).stem)
        candidates.append(str(key))
        # 若 key 为纯数字（如 predictions.json 使用整型 image_id），尝试用整数到零填充基名的映射匹配
        if str(key).isdigit():
            try:
                k_int = int(str(key))
                base = int_to_base.get(k_int, None)
                if base:
                    candidates.insert(0, base)
            except Exception:
                pass
    for b in candidates:
        if b in base_to_pair:
            return base_to_pair[b]
    # 未找到则返回空
    return (None, None)


def preview(pred_dir: Path, data_root: Path, start_index: int = 0, conf_thresh: float = 0.5) -> None:
    """
    交互预览：从 `pred_dir/predictions.json` 读取预测，再从 `data_root` 查找 RGB/IR 图片并绘制。
    键盘交互：左右方向键切换，`q`/`ESC` 退出。
    """
    pred_json = pred_dir / "predictions.json"
    groups, keys = _group_predictions(pred_json)

    # 定位 RGB/IR 图像目录
    rgb_dir, ir_dir = _find_img_dirs(data_root)
    if rgb_dir is None and ir_dir is None:
        raise FileNotFoundError(f"未在数据根目录找到图像子目录：{data_root.as_posix()}")
    base_to_pair, int_to_base = _build_pair_index(rgb_dir, ir_dir)

    # 读取类别映射（用于显示类名），若不存在则退化为显示类别数字
    id2name = _read_classes_map(data_root)

    # 窗口准备
    cv2.namedWindow("inference", cv2.WINDOW_NORMAL)
    i = max(0, int(start_index)) % len(keys)

    while True:
        k = keys[i]
        preds = groups.get(k, [])
        rgb_path, ir_path = _resolve_pair_for_key(k, base_to_pair, int_to_base)
        rgb_img = _ensure_image(rgb_path)
        ir_img = _ensure_image(ir_path)

        # 收集满足置信度阈值的 poly 与标签
        polys = []
        labels = []
        for d in preds:
            score = float(d.get("score", 0.0))
            if score < float(conf_thresh):
                continue
            poly = d.get("poly", None)
            if isinstance(poly, list) and len(poly) == 8:
                polys.append(np.array(poly, dtype=np.float32))
                cls_id = int(d.get("category_id", -1))
                name = id2name.get(cls_id, str(cls_id))
                labels.append(name)
        polys = np.array(polys, dtype=np.float32) if polys else np.zeros((0, 8), dtype=np.float32)

        # 绘制到左右两张图
        rgb_vis = _draw_polys(rgb_img, polys, labels, (0, 255, 0))
        ir_vis = _draw_polys(ir_img, polys, labels, (0, 255, 0))
        canvas = _resize_pair_left_right(rgb_vis, ir_vis)

        title = f"predictions: {pred_dir.name} | key={k} | 左=RGB 右=IR | 满足阈值={len(polys)}"
        cv2.setWindowTitle("inference", title)
        cv2.imshow("inference", canvas)

        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27):
            break
        if key in (81, 2424832):  # 左箭头
            i = (i - 1) % len(keys)
        elif key in (83, 2555904):  # 右箭头
            i = (i + 1) % len(keys)
        else:
            i = (i + 1) % len(keys)

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=str, required=True, help="包含 predictions.json 的目录")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(ROOT / "data" / "test"),
        help="数据根目录（默认 data/test）"
    )
    parser.add_argument("--start-index", type=int, default=0, help="起始样本下标（默认 0）")
    parser.add_argument("--conf", type=float, default=0.5, help="绘制置信度过滤阈值，默认 0.5")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    data_root = Path(args.data_root)
    preview(pred_dir=pred_dir, data_root=data_root, start_index=args.start_index, conf_thresh=float(args.conf))


if __name__ == "__main__":
    main()
