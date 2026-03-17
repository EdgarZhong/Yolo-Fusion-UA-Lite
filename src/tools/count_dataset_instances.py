import argparse
import ast
from pathlib import Path


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1].strip()
    return value


def _parse_dataset_yaml(yaml_path: Path):
    raw = {}
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if ":" not in s:
            continue
        key, value = s.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"path", "train", "val", "test", "names", "nc"}:
            raw[key] = value

    root = Path(_strip_quotes(raw.get("path", "")))

    def resolve(sub_path: str | None) -> Path | None:
        if not sub_path:
            return None
        sub_path = _strip_quotes(sub_path)
        p = Path(sub_path)
        return p if p.is_absolute() else root / p

    names = []
    if "names" in raw:
        try:
            names = ast.literal_eval(raw["names"])
        except Exception:
            names = []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not isinstance(names, list):
        names = []

    nc = None
    if "nc" in raw:
        try:
            nc = int(_strip_quotes(str(raw["nc"])))
        except Exception:
            nc = None
    if nc is None:
        nc = len(names)

    splits = {
        "train": resolve(raw.get("train")),
        "val": resolve(raw.get("val")),
        "test": resolve(raw.get("test")),
    }

    return root, splits, names, nc


def _infer_label_dir(img_dir: Path) -> Path:
    parent = img_dir.parent
    return parent / f"{parent.name}labels_yolo_obb"


def _count_labels(label_dir: Path, num_classes: int):
    counts = [0] * num_classes
    files = 0
    for label_file in label_dir.rglob("*.txt"):
        files += 1
        for line in label_file.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if not parts:
                continue
            try:
                cls = int(float(parts[0]))
            except Exception:
                continue
            if 0 <= cls < num_classes:
                counts[cls] += 1
    return counts, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="src/cfg/datasets/dual_obb_dronevehicle.yaml",
        help="数据集配置文件路径",
    )
    args = parser.parse_args()

    yaml_path = Path(args.data)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"未找到数据集配置文件：{yaml_path}")

    root, splits, names, num_classes = _parse_dataset_yaml(yaml_path)
    if not names:
        names = [f"class_{i}" for i in range(num_classes)]

    print(f"配置文件: {yaml_path}")
    print(f"数据集根目录: {root}")

    for split_name, img_dir in splits.items():
        if img_dir is None:
            print(f"\n[{split_name}] 未配置路径")
            continue
        label_dir = _infer_label_dir(img_dir)
        if not label_dir.exists():
            print(f"\n[{split_name}] 标签目录不存在: {label_dir}")
            continue
        counts, files = _count_labels(label_dir, num_classes)
        total = sum(counts)
        print(f"\n[{split_name}]")
        print(f"标签目录: {label_dir}")
        print(f"标签文件数: {files}")
        print(f"实例总数: {total}")
        for idx, name in enumerate(names):
            print(f"{idx}\t{name}\t{counts[idx]}")


if __name__ == "__main__":
    main()
