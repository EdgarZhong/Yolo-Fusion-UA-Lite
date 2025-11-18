import os
import sys
import glob
import shutil

def list_subsets(data_root):
    subs = []
    for s in ("train", "val", "test"):
        p = os.path.join(data_root, s)
        if os.path.isdir(p):
            subs.append(p)
    return subs

def find_dirs(sub_root):
    img_dirs = []
    label = None
    labelr = None
    yolo = None
    base = os.path.basename(sub_root).lower()
    for name in os.listdir(sub_root):
        p = os.path.join(sub_root, name)
        if not os.path.isdir(p):
            continue
        ln = name.lower()
        if "img" in ln:
            img_dirs.append(p)
        if ln.endswith("labelr"):
            labelr = p
        elif ln.endswith("label"):
            label = p
        if base == "train" and ln == "trainlabels_yolo_obb":
            yolo = p
        if base == "val" and ln == "vallabels_yolo_obb":
            yolo = p
        if base == "test" and ln == "testlabels_yolo_obb":
            yolo = p
    return img_dirs, label, labelr, yolo

def count_dataset(sub_root):
    img_dirs, label, labelr, yolo = find_dirs(sub_root)
    img_count = 0
    for d in img_dirs:
        img_count += len(glob.glob(os.path.join(d, "*.jpg")))
        img_count += len(glob.glob(os.path.join(d, "*.jpeg")))
        img_count += len(glob.glob(os.path.join(d, "*.png")))
    xml_count = 0
    if label and os.path.isdir(label):
        xml_count += len(glob.glob(os.path.join(label, "*.xml")))
    if labelr and os.path.isdir(labelr):
        xml_count += len(glob.glob(os.path.join(labelr, "*.xml")))
    yolo_count = 0
    if yolo and os.path.isdir(yolo):
        yolo_count += len(glob.glob(os.path.join(yolo, "*.txt")))
    return img_count, xml_count, yolo_count

def build_delete_list(sub_root, bases):
    img_dirs, label, labelr, yolo = find_dirs(sub_root)
    files = []
    for b in bases:
        bn = str(b)
        # images
        for d in img_dirs:
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(d, bn + ext)
                if os.path.isfile(p):
                    files.append(p)
        # xmls
        if label:
            p = os.path.join(label, bn + ".xml")
            if os.path.isfile(p):
                files.append(p)
        if labelr:
            p = os.path.join(labelr, bn + ".xml")
            if os.path.isfile(p):
                files.append(p)
        # yolo obb
        if yolo:
            p = os.path.join(yolo, bn + ".txt")
            if os.path.isfile(p):
                files.append(p)
    return sorted(set(files))

def read_mismatch(path):
    by_subset = {}
    if not os.path.isfile(path):
        return by_subset
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            subset, base = parts[0], parts[1]
            by_subset.setdefault(subset, set()).add(base)
    return by_subset

def remove_files(paths):
    removed = 0
    for p in paths:
        try:
            os.remove(p)
            removed += 1
        except FileNotFoundError:
            pass
        except PermissionError:
            try:
                os.chmod(p, 0o666)
                os.remove(p)
                removed += 1
            except Exception:
                print(f"无法删除: {p}")
        except Exception:
            print(f"删除失败: {p}")
    return removed

def main():
    data_root = os.path.join(os.path.dirname(__file__), "data")
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    mismatch_path = os.path.join(data_root, "mismatch_obb.txt")
    mis = read_mismatch(mismatch_path)
    if not mis:
        print("未发现 mismatch_obb.txt 或内容为空")
        return
    print("mismatch 概览：")
    for subset, bases in mis.items():
        print(f"  {subset}: {len(bases)} 条")
    print("示例前5条：")
    shown = 0
    for subset, bases in mis.items():
        for b in sorted(list(bases))[:5]:
            print(f"  {subset} {b}")
            shown += 1
        if shown >= 5:
            break

    before = {}
    for sub in list_subsets(data_root):
        before[os.path.basename(sub)] = count_dataset(sub)
    to_delete = []
    for subset, bases in mis.items():
        sub_root = os.path.join(data_root, subset)
        to_delete += build_delete_list(sub_root, bases)
    print(f"计划删除文件数: {len(to_delete)}")
    removed = remove_files(to_delete)
    print(f"实际删除文件数: {removed}")
    after = {}
    for sub in list_subsets(data_root):
        after[os.path.basename(sub)] = count_dataset(sub)
    print("规模统计（图片/原XML/YOLO-OBB txt）")
    for name in ("train", "val", "test"):
        b = before.get(name, (0,0,0))
        a = after.get(name, (0,0,0))
        print(f"  {name}: {b} -> {a}")

if __name__ == "__main__":
    main()