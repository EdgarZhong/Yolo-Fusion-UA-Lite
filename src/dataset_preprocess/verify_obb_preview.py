import os
import sys
import glob
import zipfile
import math
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import ast

def find_subset_dirs(sub_root):
    imgs = []
    labels = []
    for name in os.listdir(sub_root):
        p = os.path.join(sub_root, name)
        if not os.path.isdir(p):
            continue
        ln = name.lower()
        if "img" in ln:
            imgs.append(p)
        if "label" in ln:
            labels.append(p)
    return sorted(imgs), sorted(labels)

def pick_img_dirs(img_dirs):
    if not img_dirs:
        return None, None
    if len(img_dirs) == 1:
        return img_dirs[0], img_dirs[0]
    return img_dirs[0], img_dirs[1]

def pick_label_dirs(label_dirs):
    rgb = None
    ir = None
    for d in label_dirs:
        dn = os.path.basename(d).lower()
        if dn.endswith("labelr"):
            ir = d
        elif dn.endswith("label"):
            rgb = d
    if rgb is None and label_dirs:
        rgb = label_dirs[0]
    return rgb, ir

def list_images(dirp):
    if not dirp or not os.path.isdir(dirp):
        return []
    xs = []
    xs.extend(glob.glob(os.path.join(dirp, "*.jpg")))
    xs.extend(glob.glob(os.path.join(dirp, "*.jpeg")))
    xs.extend(glob.glob(os.path.join(dirp, "*.png")))
    return sorted(xs)

def read_classes_map(data_root):
    """
    读取类别映射：优先读取数据根目录下的 `classes.txt`（格式：`<id> <name>`），
    若不存在或为空，则回退读取仓库数据集 YAML（`src/cfg/datasets/dual_obb_dronevehicle.yaml`）中的 names 列表。
    """
    id2name = {}
    # 优先尝试 classes.txt
    p = os.path.join(data_root, "classes.txt")
    if os.path.isfile(p):
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
    if id2name:
        return id2name

    # 回退：读取仓库 YAML 的 names 列表
    try:
        ROOT = Path(__file__).resolve().parents[2]
        yaml_path = ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml"
        if yaml_path.is_file():
            txt = yaml_path.read_text(encoding="utf-8")
            # 简单解析：找到以 names: 开头的一行，尝试将右侧的 Python 风格列表解析为 list
            for line in txt.splitlines():
                if line.strip().startswith("names:"):
                    # 形如：names: ["car", "truck", ...]
                    data = line.split(":", 1)[1].strip()
                    try:
                        arr = ast.literal_eval(data)
                    except Exception:
                        arr = []
                    if isinstance(arr, (list, tuple)):
                        return {i: str(name) for i, name in enumerate(arr)}
    except Exception:
        pass
    return id2name

def parse_xml_objs(xml_path):
    try:
        t = ET.parse(xml_path)
    except Exception:
        return []
    root = t.getroot()
    out = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if not name:
            continue
        poly = obj.find("polygon")
        if poly is not None:
            xs = [poly.findtext(f"x{i}") for i in range(1,5)]
            ys = [poly.findtext(f"y{i}") for i in range(1,5)]
            if any(x is None or y is None for x, y in zip(xs, ys)):
                continue
            pts = np.array([[float(xi), float(yi)] for xi, yi in zip(xs, ys)], dtype=np.float32)
        else:
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                xmin = float(bb.findtext("xmin"))
                ymin = float(bb.findtext("ymin"))
                xmax = float(bb.findtext("xmax"))
                ymax = float(bb.findtext("ymax"))
            except Exception:
                continue
            pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)
        out.append({"name": name, "pts": pts})
    return out

def parse_yolo_obb_from_dir(dirp, base):
    p = os.path.join(dirp, base + ".txt")
    if not os.path.isfile(p):
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                cid = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                ang = float(parts[5])
            except Exception:
                continue
            out.append({"cid": cid, "cx": cx, "cy": cy, "w": w, "h": h, "ang": ang})
    return out

def parse_yolo_obb_from_zip(zip_path, base):
    if not os.path.isfile(zip_path):
        return []
    out = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        target = None
        suffix = base + ".txt"
        for nm in zf.namelist():
            if nm.endswith(suffix):
                target = nm
                break
        if target is None:
            return []
        with zf.open(target) as f:
            for raw in f:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    cid = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    ang = float(parts[5])
                except Exception:
                    continue
                out.append({"cid": cid, "cx": cx, "cy": cy, "w": w, "h": h, "ang": ang})
    return out

def draw_original(img, objs, color=(255,0,0)):
    if img is None:
        return None
    vis = img.copy()
    for o in objs:
        pts = o["pts"].astype(np.int32)
        cv2.polylines(vis, [pts], True, color, 2)
    return vis

def draw_yolo_obb(img, dets, id2name, color=(0,255,0)):
    if img is None:
        return None
    h, w = img.shape[:2]
    vis = img.copy()
    for d in dets:
        cx = d["cx"] * w
        cy = d["cy"] * h
        ww = d["w"] * w
        hh = d["h"] * h
        ang_deg = d["ang"] * 180.0 / math.pi
        rect = ((cx, cy), (ww, hh), ang_deg)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        cv2.polylines(vis, [box], True, color, 2)
        name = id2name.get(d.get("cid", -1), str(d.get("cid", -1)))
        cx_i = int(cx)
        cy_i = int(cy)
        cv2.putText(vis, name, (max(0, cx_i-10), max(0, cy_i-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis

def ensure_image(path):
    img = cv2.imread(path) if path and os.path.isfile(path) else None
    if img is None:
        img = np.zeros((512,512,3), dtype=np.uint8)
    return img

def resize_pair_left_right(left, right):
    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    H = max(h1, h2)
    def scale(img, H):
        h, w = img.shape[:2]
        if h == H:
            return img
        r = H / h
        return cv2.resize(img, (int(w*r), H), interpolation=cv2.INTER_LINEAR)
    l2 = scale(left, H)
    r2 = scale(right, H)
    return np.concatenate([l2, r2], axis=1)

def main():
    """
    交互预览入口：
    - 新增参数选择数据根目录（支持原始 data/ 与裁切后 data_croped/）；
    - 指定子集 train/val/test 与起始样本索引，便于快速核验裁切标签的正确性。
    """
    parser = argparse.ArgumentParser(description="DroneVehicle 双模态 OBB 标签与原始/裁切图像预览")
    parser.add_argument("--data-root", type=str, default=str(Path(__file__).resolve().parents[2] / "data"), help="数据根目录：可选 data/ 或 data_croped/")
    parser.add_argument("--subset", type=str, default="train", choices=["train", "val", "test"], help="子集选择：train/val/test")
    parser.add_argument("--start-index", type=int, default=0, help="起始样本下标（默认 0）")
    args = parser.parse_args()

    data_root = args.data_root
    subset = args.subset
    # 兼容绝对路径传入：若 data_root 已包含子集层，则直接使用；否则在 data_root 下拼接 subset
    if os.path.basename(data_root).lower() in ("train", "val", "test"):
        sub_root = data_root
        data_root = os.path.dirname(sub_root)
    else:
        sub_root = os.path.join(data_root, subset)
    img_dirs, label_dirs = find_subset_dirs(sub_root)
    rgb_img_dir, ir_img_dir = pick_img_dirs(img_dirs)
    rgb_label_dir, ir_label_dir = pick_label_dirs(label_dirs)
    subset_name = os.path.basename(sub_root).lower()
    yolo_dir = None
    if subset_name == "train":
        yolo_dir = os.path.join(sub_root, "trainlabels_yolo_obb")
        yolo_zip = os.path.join(sub_root, "trainlabels_yolo_obb.zip")
    elif subset_name == "val":
        yolo_dir = os.path.join(sub_root, "vallabels_yolo_obb")
        yolo_zip = os.path.join(sub_root, "vallabels_yolo_obb.zip")
    elif subset_name == "test":
        yolo_dir = os.path.join(sub_root, "testlabels_yolo_obb")
        yolo_zip = os.path.join(sub_root, "testlabels_yolo_obb.zip")
    else:
        yolo_dir = os.path.join(sub_root, "labels_yolo_obb")
        yolo_zip = os.path.join(sub_root, "labels_yolo_obb.zip")
    id2name = read_classes_map(data_root)
    imgs_rgb = list_images(rgb_img_dir)
    imgs_ir = list_images(ir_img_dir)
    bases = set()
    for p in imgs_rgb:
        bases.add(os.path.splitext(os.path.basename(p))[0])
    for p in imgs_ir:
        bases.add(os.path.splitext(os.path.basename(p))[0])
    bases = sorted(bases)
    if not bases:
        print("未找到图像，检查子目录是否包含图像文件")
        return
    i = max(0, int(args.start_index)) % len(bases)
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    while True:
        base = bases[i]
        ext_candidates = [".jpg", ".jpeg", ".png"]
        rgb_img_path = None
        ir_img_path = None
        for ext in ext_candidates:
            p = os.path.join(rgb_img_dir or sub_root, base + ext) if rgb_img_dir else None
            if p and os.path.isfile(p):
                rgb_img_path = p
                break
        for ext in ext_candidates:
            p = os.path.join(ir_img_dir or rgb_img_dir or sub_root, base + ext) if (ir_img_dir or rgb_img_dir) else None
            if p and os.path.isfile(p):
                ir_img_path = p
                break
        rgb_img = ensure_image(rgb_img_path)
        ir_img = ensure_image(ir_img_path)
        rgb_xml = os.path.join(rgb_label_dir, base + ".xml") if rgb_label_dir else None
        ir_xml = os.path.join(ir_label_dir, base + ".xml") if ir_label_dir else None
        rgb_objs = parse_xml_objs(rgb_xml) if rgb_xml and os.path.isfile(rgb_xml) else []
        ir_objs = parse_xml_objs(ir_xml) if ir_xml and os.path.isfile(ir_xml) else []
        dets = []
        if os.path.isdir(yolo_dir):
            dets = parse_yolo_obb_from_dir(yolo_dir, base)
        elif os.path.isfile(yolo_zip):
            dets = parse_yolo_obb_from_zip(yolo_zip, base)
        rgb_vis = draw_original(rgb_img, rgb_objs, (255,0,0))
        ir_vis = draw_original(ir_img, ir_objs, (255,0,0))
        # 使用 id2name 显示类名（优先 classes.txt，回退 YAML names）
        rgb_vis = draw_yolo_obb(rgb_vis, dets, id2name, (0,255,0))
        ir_vis = draw_yolo_obb(ir_vis, dets, id2name, (0,255,0))
        canvas = resize_pair_left_right(rgb_vis, ir_vis)
        info = f"{subset_name}:{base}  左=RGB 右=IR  原=蓝 新=绿"
        cv2.setWindowTitle("preview", info)
        cv2.imshow("preview", canvas)
        k = cv2.waitKeyEx(0)
        if k in (ord('q'), 27):
            break
        if k in (81, 2424832):
            i = (i - 1) % len(bases)
        elif k in (83, 2555904):
            i = (i + 1) % len(bases)
        else:
            i = (i + 1) % len(bases)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
