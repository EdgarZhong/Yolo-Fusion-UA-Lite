"""
多模型结果汇总表生成脚本：读取 result/ 下各模型评估 CSV 与速度基准 CSV，结合训练日志计算收敛轮次，写出 `result/多模型结果汇总.csv`。

数据来源约定（严格按项目文档实施）：
- 精度数据（mAP50、mAP50-95、Recall、Precision）：来自 `result/<模型目录>/<名称>.csv`
- 推理速度（FPS）：来自 `result/benchmark_speed_subset.csv`
- 收敛轮次（达到 mAP50>=0.6 的 epoch）：来自 `models/<...>/results.csv`

输出字段：model_code, model_label, mAP50, mAP95, Recall, Precision, FPS, epochs_to_0.6, conv_eff
其中 `conv_eff = 100.0 / epochs_to_0.6`（找不到或无效则写空）
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional


# 仓库根目录
ROOT = Path(__file__).resolve().parents[2]


# 注册 7 个模型的评估与训练日志路径（与文档一致）
REGISTRY: List[Dict[str, str]] = [
    {
        "code": "M0",
        "label": "IR-YOLOv8n",
        "eval_csv": (ROOT / "result" / "IR-YOLOv8n" / "IR-YOLOv8n.csv").as_posix(),
        "train_csv": (ROOT / "models" / "IR-YOLOv8n" / "from_scrach" / "train" / "results.csv").as_posix(),
    },
    {
        "code": "M1",
        "label": "Dual-Easy-Concat",
        "eval_csv": (ROOT / "result" / "baseline-100epoch" / "baseline-100epoch.csv").as_posix(),
        "train_csv": (ROOT / "models" / "formal" / "dualbackbone-easy-obb-formal6" / "results.csv").as_posix(),
    },
    {
        "code": "M2",
        "label": "Dual-FA-Concat(without neck)",
        "eval_csv": (ROOT / "result" / "dualbackbone-FA-Concat-100epoch" / "dualbackbone-FA-Concat-100epoch.csv").as_posix(),
        "train_csv": (ROOT / "models" / "formal" / "dualbackbone-FA-Concat-obb" / "results.csv").as_posix(),
    },
    {
        "code": "M3",
        "label": "FA-Concat (Scratch)",
        "eval_csv": (ROOT / "result" / "FA-Concat-FPN-PAN-neck-100epoch" / "FA-Concat-FPN-PAN-neck-100epoch-aug-iou0.75.csv").as_posix(),
        "train_csv": (ROOT / "models" / "formal" / "FA-Concat-FPN-PAN-neck" / "results.csv").as_posix(),
    },
    {
        "code": "M4",
        "label": "CM-FA (Scratch)",
        "eval_csv": (ROOT / "result" / "CM-FA_FPN-PAN_neck" / "CM-FA_FPN-PAN_neck.csv").as_posix(),
        "train_csv": (ROOT / "models" / "formal" / "CM-FA-Concat-FPN-PAN-neck" / "results.csv").as_posix(),
    },
    {
        "code": "M5",
        "label": "FA-Concat (Tuned)",
        "eval_csv": (ROOT / "result" / "FA-Concat_FPN-PAN_tuned" / "FA-Concat_FPN-PAN_tuned.csv").as_posix(),
        "train_csv": (ROOT / "models" / "posttrain" / "FA-Concat_FPN-PAN_tuned" / "results.csv").as_posix(),
    },
    {
        "code": "M6",
        "label": "FA-Concat (Reg)",
        "eval_csv": (ROOT / "result" / "Final_Recall_640_Regularized" / "Final_Recall_640_Regularized.csv").as_posix(),
        "train_csv": (ROOT / "models" / "posttrain" / "Final_Recall_640_Regularized" / "results.csv").as_posix(),
    },
]


def read_single_value_csv(csv_path: Path, key: str) -> Optional[float]:
    """
    从测试结果 CSV 中读取整体指标单值（第一段的 metric,value）。找不到返回 None。
    """
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = row.get("metric")
                v = row.get("value")
                if not k:
                    break
                if isinstance(k, str) and k.strip() == key:
                    try:
                        return float(v)
                    except Exception:
                        return None
    except Exception:
        return None
    return None


def read_training_series(csv_path: Path, key: str) -> List[float]:
    """
    读取训练过程 CSV（results.csv）中的某列数值序列；异常返回空列表。
    注意：Ultralytics 的 results.csv 表头含有左侧空格缩进，需使用 strip 匹配列名。
    """
    if not csv_path.exists():
        return []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # 通过 strip 匹配真正的列名
            col = None
            if reader.fieldnames:
                for fn in reader.fieldnames:
                    if isinstance(fn, str) and fn.strip() == key:
                        col = fn
                        break
            seq: List[float] = []
            for row in reader:
                name = col or key
                val = row.get(name)
                if val is None:
                    # 尝试通过 strip 查找
                    found = None
                    for k in list(row.keys()):
                        if isinstance(k, str) and k.strip() == key:
                            found = row.get(k)
                            break
                    val = found
                if val is None:
                    seq.append(float("nan"))
                    continue
                try:
                    seq.append(float(val))
                except Exception:
                    seq.append(float("nan"))
            return seq
    except Exception:
        return []


def first_epoch_reach_threshold(series: List[float], thr: float) -> Optional[int]:
    """
    返回首次达到阈值的 1-based 轮次号（例如第 1 行返回 1）；找不到返回 None。
    """
    for idx, v in enumerate(series):
        try:
            if not math.isnan(float(v)) and float(v) >= thr:
                return idx + 1
        except Exception:
            continue
    return None


def read_speed_map(csv_path: Path) -> Dict[str, float]:
    """
    读取速度基准 CSV，返回 `模型代号 -> FPS` 映射，缺失写为 NaN。
    """
    m: Dict[str, float] = {}
    if not csv_path.exists():
        return m
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("model_code") or "").strip()
            fps_s = row.get("fps")
            if not code:
                continue
            try:
                fps_v = float(fps_s) if fps_s is not None and len(str(fps_s)) else float("nan")
            except Exception:
                fps_v = float("nan")
            m[code] = fps_v
    return m


def main() -> None:
    out_path = ROOT / "result" / "多模型结果汇总.csv"
    speed_path = ROOT / "result" / "benchmark_speed_subset.csv"
    speed_map = read_speed_map(speed_path)

    rows: List[List[str]] = []
    for r in REGISTRY:
        code = r["code"]
        label = r["label"]
        eval_csv = Path(r["eval_csv"])  # 精度数据从 result/ 读取
        train_csv = Path(r["train_csv"])  # 收敛从训练日志读取

        m50 = read_single_value_csv(eval_csv, "metrics/mAP50(B)")
        m95 = read_single_value_csv(eval_csv, "metrics/mAP50-95(B)")
        rec = read_single_value_csv(eval_csv, "metrics/recall(B)")
        pre = read_single_value_csv(eval_csv, "metrics/precision(B)")

        series = read_training_series(train_csv, "metrics/mAP50(B)")
        ep = first_epoch_reach_threshold(series, 0.6)
        conv_eff = (100.0 / float(ep)) if (ep is not None and ep > 0) else float("nan")
        fps = speed_map.get(code, float("nan"))

        def _fmt(x: Optional[float]) -> str:
            try:
                return f"{float(x):.5f}"
            except Exception:
                return ""

        rows.append([
            code,
            label,
            _fmt(m50),
            _fmt(m95),
            _fmt(rec),
            _fmt(pre),
            _fmt(fps),
            str(ep or ""),
            _fmt(conv_eff),
        ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_code", "model_label", "mAP50", "mAP95", "Recall", "Precision", "FPS", "epochs_to_0.6", "conv_eff"])
        for r in rows:
            w.writerow(r)

    print(f"已生成综合汇总：{out_path.as_posix()}，模型数={len(rows)}")


if __name__ == "__main__":
    main()

