"""
快速验证脚本：自动选取最新训练权重，直接跑验证集以检查验证流程正确性。

使用方法：
- 在项目根目录执行：`python src/trainning/quick_val_latest.py`
"""

from pathlib import Path
import sys
import argparse


def _find_latest_weights(base_dir: Path):
    """
    在给定根目录下递归搜索所有 `weights/last.pt` 与 `weights/best.pt`，返回最近修改的权重路径。
    """
    candidates = []
    for p in base_dir.rglob("weights/*.pt"):
        name = p.name.lower()
        if name in ("last.pt", "best.pt"):
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.stat().st_mtime)


def main():
    ROOT = Path(__file__).resolve().parents[2]
    ULTRA = ROOT / "ultralytics-8.2"
    if str(ULTRA) not in sys.path:
        sys.path.insert(0, str(ULTRA))

    from ultralytics.models.yolo.obb import OBBValidator

    # ============= 命令行参数解析：允许手动指定权重与数据配置 =============
    parser = argparse.ArgumentParser(description="使用最新或指定权重执行 OBB 验证")
    parser.add_argument("--weights", type=str, default="", help="显式指定权重文件路径（可选）")
    parser.add_argument("--data", type=str, default=str(ROOT / "src/cfg/datasets/dual_obb_dronevehicle.yaml"), help="数据集配置 YAML 路径")
    parser.add_argument("--device", type=int, default=0, help="GPU 设备编号；CPU 可设为 -1")
    parser.add_argument("--batch", type=int, default=8, help="验证批大小")
    parser.add_argument("--imgsz", type=int, default=832, help="验证图像尺寸")
    parser.add_argument("--rect", action="store_true", help="启用矩形验证（更快更稳）")
    args_ns = parser.parse_args()

    data_cfg = args_ns.data
    # 默认在 models 根目录下搜索最新权重，涵盖 baseline/formal 等所有运行目录
    if args_ns.weights:
        weights = Path(args_ns.weights)
    else:
        runs_dir = ROOT / "models"
        weights = _find_latest_weights(runs_dir)
        if weights is None:
            raise FileNotFoundError(
                f"未找到权重文件，请确认 {runs_dir.as_posix()} 下存在 weights/last.pt 或 weights/best.pt，或通过 --weights 指定"
            )

    args = {
        "task": "obb",
        "data": data_cfg,
        "imgsz": int(args_ns.imgsz),
        "split": "val",
        "device": int(args_ns.device),
        "batch": int(args_ns.batch),
        "half": False,
        "rect": bool(args_ns.rect),
        "save_json": False,
        "plots": False,
    }

    print(f"[QuickVal] 使用权重: {weights.as_posix()}")
    print(f"[QuickVal] 数据配置: {data_cfg}")

    validator = OBBValidator(args=args)
    stats = validator(model=str(weights))
    print("[QuickVal] 验证完成，统计指标：")
    print(stats)


if __name__ == "__main__":
    main()