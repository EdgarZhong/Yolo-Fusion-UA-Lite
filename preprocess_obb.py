import os
import sys
import glob
import math
import logging
import re
import xml.etree.ElementTree as ET
import numpy as np
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

def normalize_class_name(name: str) -> str:
    """Normalize class names to fix dataset typos and ensure consistency.
    Specifically unify 'feright car' and 'feright_car' to 'feright_car'.
    """
    if not name:
        return ""
    base = name.strip().lower()
    base = re.sub(r"[\s\-]+", "_", base)
    # unify freight/feright variants to 'freight_car' per project convention
    freight_variants = {"freight_car", "freightcar", "freight", "freight_vehicle"}
    feright_variants = {"feright_car", "ferightcar", "feright"}
    if base in freight_variants or base in feright_variants:
        return "freight_car"
    if base in {"car", "bus", "truck", "van", "truvk"}:
        return "truck" if base == "truvk" else base
    if base == "*":
        return ""
    return base

def list_subsets(data_root):
    """List available dataset splits under data_root."""
    subs = []
    for s in ("train", "val", "test"):
        p = os.path.join(data_root, s)
        if os.path.isdir(p):
            subs.append(p)
    logging.info(f"Found subsets: {', '.join(os.path.basename(x) for x in subs)}")
    return subs

def find_dirs(sub_root):
    """Detect image and label directories in a subset folder.
    Skip any directories already ending with '_yolo_obb'.
    """
    imgs = None
    labels = []
    for name in os.listdir(sub_root):
        p = os.path.join(sub_root, name)
        if not os.path.isdir(p):
            continue
        ln = name.lower()
        if ln.endswith("_yolo_obb"):
            continue
        if ln.endswith("img"):
            imgs = p
        if "label" in ln:
            labels.append(p)
    logging.info(f"Subset '{os.path.basename(sub_root)}' => imgs: {imgs}, labels: {labels}")
    return imgs, labels

def parse_xml_objects(xml_path):
    """Parse a VOC-style XML and return image size and a list of objects with corners and AABB.
    Supports <polygon> 4 points or <bndbox> fallback.
    Applies class name normalization.
    """
    t = ET.parse(xml_path)
    root = t.getroot()
    w = root.findtext("size/width")
    h = root.findtext("size/height")
    iw = int(w) if w is not None else None
    ih = int(h) if h is not None else None
    objs = []
    total = 0
    skipped_empty_name = 0
    skipped_poly_incomplete = 0
    skipped_no_box = 0
    for obj in root.findall("object"):
        total += 1
        raw_name = obj.findtext("name") or "object"
        name = normalize_class_name(raw_name)
        if not name:
            skipped_empty_name += 1
            continue
        poly = obj.find("polygon")
        if poly is not None:
            xs = [poly.findtext(f"x{i}") for i in range(1,5)]
            ys = [poly.findtext(f"y{i}") for i in range(1,5)]
            if any(x is None or y is None for x, y in zip(xs, ys)):
                skipped_poly_incomplete += 1
                continue
            pts = [[float(xi), float(yi)] for xi, yi in zip(xs, ys)]
        else:
            bb = obj.find("bndbox")
            if bb is not None:
                xmin = float(bb.findtext("xmin"))
                ymin = float(bb.findtext("ymin"))
                xmax = float(bb.findtext("xmax"))
                ymax = float(bb.findtext("ymax"))
                pts = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            else:
                skipped_no_box += 1
                continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        aabb = [min(xs), min(ys), max(xs), max(ys)]
        objs.append({"class": name, "corners": pts, "aabb": aabb})
    if skipped_empty_name or skipped_poly_incomplete or skipped_no_box:
        logging.info(
            f"Parsed {xml_path}: total={total}, kept={len(objs)}, "
            f"skip_empty_name={skipped_empty_name}, skip_poly_incomplete={skipped_poly_incomplete}, skip_no_box={skipped_no_box}"
        )
    return iw, ih, objs

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

def merge_by_iou(rgb_objs, ir_objs, thr=0.5):
    """Merge RGB and IR objects by AABB IoU and matching normalized class names.
    If matched IoU>=thr, combine corners (8 points) before minAreaRect.
    """
    final = []
    used = set()
    for i, ro in enumerate(rgb_objs):
        best = -1.0
        best_j = -1
        for j, io in enumerate(ir_objs):
            if j in used:
                continue
            if ro["class"] != io["class"]:
                continue
            v = iou(ro["aabb"], io["aabb"])
            if v > best:
                best = v
                best_j = j
        if best >= thr and best_j >= 0:
            pts = ro["corners"] + ir_objs[best_j]["corners"]
            final.append({"class": ro["class"], "corners": pts})
            used.add(best_j)
        else:
            final.append({"class": ro["class"], "corners": ro["corners"]})
    for j, io in enumerate(ir_objs):
        if j not in used:
            final.append({"class": io["class"], "corners": io["corners"]})
    return final

def rect_to_yolo(iw, ih, pts):
    arr = np.array(pts, dtype=np.float32)
    rect = cv2.minAreaRect(arr)
    (cx, cy), (w, h), angle_deg = rect
    cxn = cx / iw if iw else 0.0
    cyn = cy / ih if ih else 0.0
    wn = w / iw if iw else 0.0
    hn = h / ih if ih else 0.0
    angle_rad = angle_deg * math.pi / 180.0
    return cxn, cyn, wn, hn, angle_rad

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def collect_class_names(xml_dirs):
    """Collect normalized class names from XMLs across directories."""
    names = set()
    for d in xml_dirs:
        xlist = glob.glob(os.path.join(d, "*.xml"))
        for xp in xlist:
            try:
                _, _, objs = parse_xml_objects(xp)
            except Exception as e:
                logging.warning(f"Failed parsing classes from {xp}: {e}")
                continue
            for o in objs:
                names.add(o["class"].strip())
    # Filter to known classes if desired
    names = {n for n in names if n}
    logging.info(f"Collected classes: {sorted(names)}")
    return sorted(names)

def build_id_map(names):
    return {n: i for i, n in enumerate(names)}

def write_classes(data_root, id_map):
    """Write class id-name mapping to classes.txt."""
    p = os.path.join(data_root, "classes.txt")
    with open(p, "w", encoding="utf-8") as f:
        for n, i in sorted(id_map.items(), key=lambda x: x[1]):
            f.write(f"{i} {n}\n")
    logging.info(f"Wrote classes to {p}: {len(id_map)} classes")

def get_image_size(img_path):
    """Read image to obtain width/height if XML lacks size."""
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    return w, h

def process_subset(sub_root, id_map):
    """Process one subset: iterate images, pair RGB/IR XMLs, write single labels directory."""
    imgs_dir, label_dirs = find_dirs(sub_root)
    if not label_dirs:
        logging.warning(f"No label directories found under {sub_root}")
        return
    # Identify primary and secondary label dirs (e.g., *_label and *_labelr)
    rgb_dir = None
    ir_dir = None
    for d in sorted(label_dirs):
        dn = os.path.basename(d).lower()
        if dn.endswith("labelr"):
            ir_dir = d
        elif dn.endswith("label"):
            rgb_dir = d
    # Fallback: if only one label dir, treat as rgb_dir
    if rgb_dir is None and label_dirs:
        rgb_dir = label_dirs[0]
    subset_name = os.path.basename(sub_root).lower()
    if subset_name == "train":
        out_dir = os.path.join(sub_root, "trainlabels_yolo_obb")
    elif subset_name == "val":
        out_dir = os.path.join(sub_root, "vallabels_yolo_obb")
    elif subset_name == "test":
        out_dir = os.path.join(sub_root, "testlabels_yolo_obb")
    else:
        out_dir = os.path.join(sub_root, "labels_yolo_obb")
    ensure_dir(out_dir)
    total_written = 0
    images = []
    if imgs_dir and os.path.isdir(imgs_dir):
        images = sorted(glob.glob(os.path.join(imgs_dir, "*.jpg")))
        images += sorted(glob.glob(os.path.join(imgs_dir, "*.png")))
    else:
        logging.warning(f"No image directory for {sub_root}, pairing by XML base names")
        # Pair by union of XML base names
        union_bases = set()
        for d in [rgb_dir, ir_dir] if ir_dir else [rgb_dir]:
            if d:
                for xp in glob.glob(os.path.join(d, "*.xml")):
                    union_bases.add(os.path.splitext(os.path.basename(xp))[0])
        images = [os.path.join(sub_root, "dummy", b + ".jpg") for b in sorted(union_bases)]
    for imgp in images:
        bn = os.path.splitext(os.path.basename(imgp))[0]
        rgb_xml = os.path.join(rgb_dir, bn + ".xml") if rgb_dir else None
        ir_xml = os.path.join(ir_dir, bn + ".xml") if ir_dir else None
        rgb_objs = []
        ir_objs = []
        iw = ih = None
        if rgb_xml and os.path.isfile(rgb_xml):
            try:
                iw, ih, rgb_objs = parse_xml_objects(rgb_xml)
            except Exception as e:
                logging.warning(f"Failed parsing {rgb_xml}: {e}")
        if ir_xml and os.path.isfile(ir_xml):
            try:
                iw2, ih2, ir_objs = parse_xml_objects(ir_xml)
                iw = iw or iw2
                ih = ih or ih2
            except Exception as e:
                logging.warning(f"Failed parsing {ir_xml}: {e}")
        if iw is None or ih is None:
            iw, ih = get_image_size(imgp)
        final = []
        if rgb_objs or ir_objs:
            if ir_dir:
                final = merge_by_iou(rgb_objs, ir_objs, thr=0.5)
            else:
                final = [{"class": o["class"], "corners": o["corners"]} for o in rgb_objs]
        outp = os.path.join(out_dir, bn + ".txt")
        if not final:
            open(outp, "w").close()
            total_written += 1
            continue
        lines = []
        for o in final:
            cid = id_map.get(o["class"].strip(), 0)
            cxn, cyn, wn, hn, ang = rect_to_yolo(iw, ih, o["corners"]) if iw and ih else (0.0, 0.0, 0.0, 0.0, 0.0)
            lines.append(f"{cid} {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f} {ang:.6f}")
        with open(outp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        total_written += 1
    logging.info(f"[{os.path.basename(sub_root)}] wrote {total_written} labels to {out_dir}")

def read_classes_file(classes_path):
    """Read user-provided classes.txt as authoritative mapping name->id."""
    if not os.path.isfile(classes_path):
        raise FileNotFoundError(f"classes.txt not found: {classes_path}")
    id_map = {}
    with open(classes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            cid_str, cname = parts[0], parts[1]
            try:
                cid = int(cid_str)
            except Exception:
                continue
            id_map[cname] = cid
    logging.info(f"Loaded classes from {classes_path}: {len(id_map)} classes -> {sorted(id_map.items(), key=lambda x: x[1])}")
    return id_map

def main():
    """Entry point: read classes.txt mapping, then process each subset once."""
    data_root = os.path.join(os.path.dirname(__file__), "data")
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    logging.info(f"Data root: {data_root}")
    subs = list_subsets(data_root)
    classes_path = os.path.join(data_root, "classes.txt")
    id_map = read_classes_file(classes_path)
    for s in subs:
        process_subset(s, id_map)

if __name__ == "__main__":
    main()