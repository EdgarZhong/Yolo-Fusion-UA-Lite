"""
在测试集上全量评估最佳模型 CM-FA-Transferred，并生成混淆矩阵图片到 result/<模型目录>/。

用法示例：
- 默认权重与目录：
  - python src/tools/build_confusion_matrix_best.py --device 0
- 指定权重文件：
  - python src/tools/build_confusion_matrix_best.py --weights models/posttrain/CM-FA-Transferred/weights/best.pt --device 0
- 指定输出目录与运行名：
  - python src/tools/build_confusion_matrix_best.py --result-dir result --run-name CM-FA-Transferred

说明：
- 脚本强制 `plots=True` 以收集混淆矩阵，`save_json=True` 以便后续需要时保存预测结果。
- 输出文件位于 `result/<run_name>/`：
  - confusion_matrix.png（非归一化）
  - confusion_matrix_normalized.png（按列归一化）
  - predictions.json（若启用 `save_json=True`）
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBValidator
from ultralytics.utils.metrics import ConfusionMatrix

# 路径宏（推荐按需修改）
DATA_CFG = ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml"
DEFAULT_RESULT_DIR = ROOT / "result"
DEFAULT_RUN_NAME = "CM-FA-Transferred"
DEFAULT_WEIGHTS_DIR = ROOT / "models" / "posttrain" / "CM-FA-Transferred" / "weights"


def _pick_weights(user_path: str | None) -> Path:
    """
    选择权重文件：优先使用用户传入路径；否则在默认目录优先选择 best.pt，其次 last.pt。
    """
    if user_path:
        p = Path(user_path)
        if p.is_dir():
            for name in ("best.pt", "last.pt"):
                cand = p / name
                if cand.exists():
                    return cand
        return p
    for name in ("best.pt", "last.pt"):
        cand = DEFAULT_WEIGHTS_DIR / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"未找到权重文件：{DEFAULT_WEIGHTS_DIR}/best.pt 或 last.pt，请通过 --weights 指定")


def run_eval_and_confmat(
    weights: Path,
    device: str = "0",
    data_cfg: Path = DATA_CFG,
    result_dir: Path | None = None,
    run_name: str = DEFAULT_RUN_NAME,
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.25,
    iou: float = 0.75,
    max_det: int = 500,
    test_aug: bool = False,
):
    """
    在测试集上执行 OBB 验证并生成混淆矩阵图片。
    """
    save_root = Path(result_dir) if result_dir else DEFAULT_RESULT_DIR
    save_root.mkdir(parents=True, exist_ok=True)
    save_dir = save_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    workers = 0 if str(device).lower() == "cpu" else 2  # 根据设备类型设置并发线程数（CPU 场景置 0 更稳定）

    args = dict(
        task="obb",
        model=str(weights),
        data=str(data_cfg),
        split="test",
        imgsz=imgsz,
        rect=True,
        batch=batch,
        workers=workers,
        device=device,
        plots=True,
        save_json=True,
        conf=float(conf),
        iou=float(iou),
        augment=bool(test_aug),  # 与统一测试脚本保持一致：默认不启用测试时增强（可通过 CLI 打开）
        max_det=int(max_det),
        project=str(save_root),
        name=run_name,
    )

    validator = OBBValidator(args=args)
    stats = validator(model=str(weights))

    cm = getattr(validator.metrics, "confusion_matrix", None)
    if isinstance(cm, ConfusionMatrix):
        cm.plot(normalize=False, save_dir=str(save_dir), names=getattr(validator, "names", ()))
        cm.plot(normalize=True, save_dir=str(save_dir), names=getattr(validator, "names", ()))
        print(f"已保存混淆矩阵到：{save_dir}")
    else:
        print("未获取到混淆矩阵，请检查参数 plots=True 是否生效")

    return save_dir, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="运行设备：'cpu' 或 GPU 索引")
    parser.add_argument("--weights", type=str, default="", help="权重文件或目录（为空则使用默认最佳模型路径）")
    parser.add_argument("--result-dir", type=str, default=str(DEFAULT_RESULT_DIR), help="结果根目录")
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME, help="结果子目录名（模型目录名）")
    parser.add_argument("--imgsz", type=int, default=640, help="测试图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="测试批大小")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认 0.25）")
    parser.add_argument("--iou", type=float, default=0.75, help="NMS的IOU阈值")
    parser.add_argument("--max-det", type=int, default=500, help="每图最大检测数量")
    parser.add_argument("--test-aug", action="store_true", help="启用测试时增强（多尺度/翻转）")
    parser.add_argument("--data-cfg", type=str, default=str(DATA_CFG), help="数据集 YAML 路径")
    args = parser.parse_args()

    w = _pick_weights(args.weights or None)
    out_dir, stats = run_eval_and_confmat(
        weights=w,
        device=args.device,
        data_cfg=Path(args.data_cfg),
        result_dir=Path(args.result_dir),
        run_name=args.run_name,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=float(args.conf),
        iou=args.iou,
        max_det=args.max_det,
        test_aug=bool(args.test_aug),
    )
    print(f"评估完成，统计指标：{stats}\n输出目录：{out_dir}")


if __name__ == "__main__":
    main()
