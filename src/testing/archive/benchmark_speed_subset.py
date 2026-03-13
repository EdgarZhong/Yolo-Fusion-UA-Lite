"""
推理速度基准测试（子集版）：在测试集前 N 组样本上，统一 640px 输入，测量 OBB 模型的推理速度（含预处理+推理+NMS）。

实现要求（与《多模型性能评估与可视化实施文档.md》一致）：
- 脚本路径：src/testing/benchmark_speed_subset.py
- 使用测试集的前 1000 组数据（可通过 --limit 参数覆盖），统一输入 640px。
- 脚本顶部注册 7 个模型的权重路径，单次运行依次完成所有模型的速度测试。
- 输出综合 CSV，列出每个模型的 Preprocess/Inference/Postprocess 三阶段耗时（ms/img）及 FPS。

注意：
- 本脚本仅度量评估环节中的预处理、模型前向推理以及后处理（NMS 等）耗时，不包含数据加载与磁盘 IO 的时间，
  与 Ultralytics Validator 的规范保持一致，便于不同机器之间可比。
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple

import torch

# ====== 保证导入本仓库随附的 ultralytics-8.2 源码 ======
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBValidator  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.data import build_yolo_dataset  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


# ===================== 路径与常量约定 =====================
# 统一使用裁切后 640×512 的双模态数据集配置
DATA_CFG = ROOT / "src/cfg/datasets/dual_obb_dronevehicle.yaml"

# 统一测试标准参数（可通过 CLI 覆盖）
IMG_SIZE = 640
BATCH_DEFAULT = 16
CONF_THRES_DEFAULT = 0.25
IOU_THRES_DEFAULT = 0.75
MAX_DET_DEFAULT = 1000


# ===================== 模型注册表（权重路径集合） =====================
# 说明：
# - 所有路径以仓库根目录为基准；若 best.pt 不存在将回退到 last.pt；仍不存在则报错。
# - code/label 与评估文档保持一致，便于后续整合到雷达图。
MODEL_REGISTRY: List[Dict[str, str]] = [
    {
        "code": "M0",
        "label": "IR-YOLOv8n",
        "base": (ROOT / "models" / "IR-YOLOv8n" / "from_scrach" / "train").as_posix(),
        "modal": "ir",
    },
    {
        "code": "M1",
        "label": "Dual-Easy-Concat",
        "base": (ROOT / "models" / "formal" / "dualbackbone-easy-obb-formal6").as_posix(),
        "modal": "dual",
    },
    {
        "code": "M2",
        "label": "Dual-FA-Concat(without neck)",
        "base": (ROOT / "models" / "formal" / "dualbackbone-FA-Concat-obb").as_posix(),
        "modal": "dual",
    },
    {
        "code": "M3",
        "label": "FA-Concat (Scratch)",
        "base": (ROOT / "models" / "formal" / "FA-Concat-FPN-PAN-neck").as_posix(),
        "modal": "dual",
    },
    {
        "code": "M4",
        "label": "CM-FA (Scratch)",
        "base": (ROOT / "models" / "formal" / "CM-FA-Concat-FPN-PAN-neck").as_posix(),
        "modal": "dual",
    },
    {
        "code": "M5",
        "label": "FA-Concat (Tuned)",
        "base": (ROOT / "models" / "posttrain" / "FA-Concat_FPN-PAN_tuned").as_posix(),
        "modal": "dual",
    },
    {
        "code": "M6",
        "label": "FA-Concat (Reg)",
        "base": (ROOT / "models" / "posttrain" / "Final_Recall_640_Regularized").as_posix(),
        "modal": "dual",
    },
]


def _find_weights(base: Path) -> Path:
    """
    在给定的训练输出目录下寻找权重文件。

    选择优先级：weights/best.pt > weights/last.pt > 其它 *.pt（任选其一）。
    若不存在任何权重文件，则抛出 FileNotFoundError。
    """
    wdir = base / "weights"
    candidates = [wdir / "best.pt", wdir / "last.pt"]
    for p in candidates:
        if p.exists():
            return p
    other = next(iter(wdir.glob("*.pt")), None)
    if other is None:
        raise FileNotFoundError(f"未找到权重文件：{wdir.as_posix()} 下不存在 best.pt/last.pt")
    return other


def build_subset_loader(batch_size: int, workers: int, limit: int, modal: str = "dual") -> torch.utils.data.DataLoader:
    """
    构建测试集前 `limit` 个样本的 DataLoader（用于速度评估）。

    实现细节：
    - 使用 `check_det_dataset` 读取 YAML，并通过 `build_yolo_dataset(cfg, data['test'], ...)` 构造 OBB 模式的数据集；
    - 通过 `torch.utils.data.Subset` 截取前 `limit` 个样本；
    - collate_fn 复用数据集自带的拼接函数，关闭 pin_memory 与 shuffle；
    - 返回可直接传入 Validator 的 DataLoader。
    """
    data = check_det_dataset(str(DATA_CFG))
    cfg = get_cfg(overrides=dict(task="obb", imgsz=IMG_SIZE, rect=True))
    # 根据 modal 决定加载的目录：dual -> testimg；ir -> testimgr；rgb -> 仍用 testimg（但会按 3 通道处理）
    test_path = data.get("test")
    if isinstance(test_path, (str, Path)):
        tp = Path(test_path)
    else:
        tp = Path(str(test_path))
    if str(modal).lower() == "ir":
        # 将末级目录名从 *img 替换为 *imgr（IR 单模态）
        leaf = tp.name
        if leaf.endswith("img"):
            tp = tp.parent / (leaf + "r")
        else:
            # 兜底：直接拼接同级 testimgr
            tp = tp.parent / "testimgr"
    # rgb/dual 情况下都传入原始 testimg，让构造函数自行选择 YOLODualDataset 或 YOLODataset
    base_ds = build_yolo_dataset(cfg, tp.as_posix(), batch_size, data, mode="val")

    # 计算子集大小，至少为 1；若 limit 大于数据集长度则自动裁剪到最大长度
    total = len(base_ds)
    k = max(1, min(int(limit), total))

    from torch.utils.data import DataLoader, Subset

    indices = list(range(k))
    subset = Subset(base_ds, indices)
    loader = DataLoader(
        dataset=subset,
        batch_size=min(batch_size, len(indices)),
        shuffle=False,
        num_workers=workers,
        collate_fn=getattr(base_ds, "collate_fn", None),
        pin_memory=False,
    )
    return loader


def bench_one(weights: Path, device: str, batch: int, workers: int, conf: float, iou: float, max_det: int, augment: bool,
              loader: Optional[torch.utils.data.DataLoader], split: str = "test") -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    对单个模型执行速度基准测试，返回：
    - speed_ms：每张图片的三个阶段耗时（ms/img）：preprocess、inference、postprocess
    - extra：包含 avg_total_ms 与 fps 的综合指标

    说明：
    - 若传入 `loader` 则使用子集 DataLoader；否则由 Validator 内部按 YAML 自动构建全量测试集。
    - 速度统计来自 Validator 的 `Profile` 聚合结果，不包含数据加载时间，便于跨机器比较。
    """
    args = dict(
        task="obb",
        model=str(weights),
        data=str(DATA_CFG),
        split=str(split),
        imgsz=IMG_SIZE,
        rect=True,
        batch=batch,
        workers=workers,
        device=device,
        plots=False,
        save_json=False,
        conf=float(conf),
        iou=float(iou),
        augment=bool(augment),
        max_det=int(max_det),
    )

    validator = OBBValidator(dataloader=loader, args=args) if loader is not None else OBBValidator(args=args)
    print(f"[SpeedBench] 已创建验证器，max_det={getattr(validator.args, 'max_det', None)}，imgsz={IMG_SIZE}，batch={batch}")

    # 执行验证以触发速度统计；返回的 stats 为精度相关指标，此处不使用
    _ = validator(model=str(weights))

    # validator.speed 为各阶段的 ms/img 数值
    speed_ms = {k: float(v) for k, v in validator.speed.items()}
    # 非训练模式下 loss 阶段时间恒为 0，可忽略
    avg_total_ms = speed_ms.get("preprocess", 0.0) + speed_ms.get("inference", 0.0) + speed_ms.get("postprocess", 0.0)
    fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0

    extra = {"avg_total_ms": avg_total_ms, "fps": fps}
    return speed_ms, extra


def main() -> None:
    """
    主入口：
    - 读取 7 个模型的权重路径，依次在测试集前 `--limit` 个样本上进行速度基准测试；
    - 将结果写入综合 CSV：`result/benchmark_speed_subset.csv`，便于后续整合到《多模型结果汇总.csv》。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="运行设备：cpu 或 GPU 索引，如 0/1")
    parser.add_argument("--batch", type=int, default=BATCH_DEFAULT, help="批大小（受显存/CPU限制影响）")
    parser.add_argument("--workers", type=int, default=-1, help="DataLoader 线程数（-1 表示自动：CPU=0，否则=2）")
    parser.add_argument("--conf", type=float, default=CONF_THRES_DEFAULT, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=IOU_THRES_DEFAULT, help="NMS 的 IoU 阈值")
    parser.add_argument("--max-det", type=int, default=MAX_DET_DEFAULT, help="最大候选框数量")
    parser.add_argument("--augment", action="store_true", help="启用测试时增强（模拟非常规推理，默认关闭）")
    parser.add_argument("--limit", type=int, default=1000, help="使用测试集前 N 个样本进行速度评估")
    parser.add_argument("--output", type=str, default=str((ROOT / "result" / "benchmark_speed_subset.csv")), help="综合 CSV 输出路径")
    args = parser.parse_args()

    device = select_device(args.device)
    # workers 自动判定：CPU 场景下置 0；GPU 场景给 2 以兼顾 IO 与稳定性
    workers = (0 if device.type == "cpu" else 2) if int(args.workers) < 0 else int(args.workers)

    # 构建子集 DataLoader（前 N 个样本），统一 imgsz=640、rect=True
    # 注意：M0 为单模态 IR，其他为双模态；据此构建各自的子集 DataLoader
    # 为提高整体运行效率，这里复用一个 DataLoader 仅供双模态模型，IR 模型单独构建一次
    dual_loader = build_subset_loader(batch_size=max(1, int(args.batch)), workers=workers, limit=max(1, int(args.limit)), modal="dual")
    ir_loader = build_subset_loader(batch_size=max(1, int(args.batch)), workers=workers, limit=max(1, int(args.limit)), modal="ir")

    # 逐模型执行速度评估并收集结果
    rows: List[List[str]] = []
    for item in MODEL_REGISTRY:
        code = item["code"]
        label = item["label"]
        base = Path(item["base"])
        try:
            weights = _find_weights(base)
        except Exception as e:
            print(f"[SpeedBench][{code}] 跳过：{e}")
            continue

        print(f"[SpeedBench][{code}] 开始：{label} -> {weights.as_posix()}")
        # 针对单模态 IR 模型使用 ir_loader，其余模型使用 dual_loader
        use_loader = ir_loader if item.get("modal", "dual").lower() == "ir" else dual_loader
        split = "ir" if item.get("modal", "dual").lower() == "ir" else "test"
        speed_ms, extra = bench_one(
            weights=weights,
            device=args.device,
            batch=max(1, int(args.batch)),
            workers=workers,
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
            augment=bool(args.augment),
            loader=use_loader,
            split=split,
        )

        rows.append([
            code,
            label,
            weights.as_posix(),
            f"{speed_ms.get('preprocess', 0.0):.4f}",
            f"{speed_ms.get('inference', 0.0):.4f}",
            f"{speed_ms.get('postprocess', 0.0):.4f}",
            f"{extra.get('avg_total_ms', 0.0):.4f}",
            f"{extra.get('fps', 0.0):.5f}",
        ])

    # 写出综合 CSV（若无任何结果将仍然创建空表，便于后续数据整合脚本判断）
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_code", "model_label", "weights",
                    "preprocess_ms_per_img", "inference_ms_per_img", "postprocess_ms_per_img",
                    "avg_total_ms_per_img", "fps"])
        for r in rows:
            w.writerow(r)

    print(f"[SpeedBench] 已保存综合 CSV：{out_path.as_posix()}，模型数={len(rows)}，limit={args.limit}，imgsz={IMG_SIZE}")


if __name__ == "__main__":
    main()
