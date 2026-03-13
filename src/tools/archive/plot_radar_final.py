"""
雷达图可视化脚本（最终综合评估）：根据“多模型性能评估与可视化实施文档.md”实现

功能概述：
- 读取各模型在测试集的精度指标（来自 `result/<模型目录>/<名称>.csv`）
- 读取训练过程的 `results.csv`，计算“收敛效率”（100 / 达到 mAP50>=0.6 的轮次）
- 汇总 6 个维度：mAP50、mAP50-95、Recall、Precision、FPS（可选）、Convergence Efficiency
- 绘制 7 个模型的最终雷达图，颜色分配与 `plot_training.py` 保持一致的配色池

更新说明（取消归一化）：
- 取消“跨模型的最小-最大归一化（min-max）”，改为按自然量纲设置比例尺：
  - mAP/Recall/Precision 使用原始 0–1 数值直接绘制（半径刻度 0–1）。
  - FPS 使用结果汇总的最大值的“漂亮上限”（10 的倍数，含 5% 裁剪冗余）作为比例尺，将实际 FPS 映射到 0–1 半径，并在轴标签中标注上限。
  - Convergence（收敛效率）同样采用结果汇总的“漂亮上限”作为比例尺并在轴标签中标注。
- 若某维度缺失值则按 0 处理，不再说明“归一化为 0”。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager
import os
import webbrowser


# ===================== 路径与常量 =====================
ROOT = Path(__file__).resolve().parents[2]  # 仓库根目录 YOLO-Fusion-UA-Lite/

# 与 plot_training.py 保持一致的配色池（顺序固定）
COLOR_POOL: List[str] = [
    "#4e79a7",  # 蓝
    "#e15759",  # 红
    "#76b7b2",  # 青
    "#f28e2b",  # 橙
    "#59a14f",  # 绿
    "#edc948",  # 黄
    "#b07aa1",  # 紫
    "#ff9da7",  # 粉
    "#9c755f",  # 棕
    "#bab0ab",  # 灰
]


def setup_cn_font() -> str:
    """
    设置 Matplotlib 中文字体，避免中文乱码；返回实际使用的字体名称（若未找到则返回空字符串）。
    """
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "NSimSun",
        "KaiTi",
        "FangSong",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = candidates + list(rcParams.get("font.sans-serif", []))
    rcParams["axes.unicode_minus"] = False
    for name in candidates:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            if isinstance(path, str) and len(path):
                return name
        except Exception:
            continue
    return ""


def read_single_value_csv(csv_path: Path, key: str) -> Optional[float]:
    """
    从测试结果 CSV（如 `result/<模型>/<名称>.csv`）中读取某个最终指标的单值。
    约定：该 CSV 为“最终评估结果”，包含若干行，第一段为整体指标（metric,value），后续为 per-class；
    我们读取第一段里的指定 key（例如 'metrics/mAP50(B)'）。
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
                    # 读取到第二段（per-class）或格式不同，停止搜索
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
    从训练过程 CSV（models/<...>/results.csv）中读取某列的数值序列。
    若缺失或格式异常，返回空列表。
    """
    if not csv_path.exists():
        return []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            seq: List[float] = []
            for row in reader:
                val = row.get(key)
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
    返回首次达到阈值的“轮次号”（基于行序，起始为 0 表示第 1 轮），找不到则返回 None。
    """
    for idx, v in enumerate(series):
        try:
            if not math.isnan(float(v)) and float(v) >= thr:
                # 将 0-based 行号转换为 1-based epoch 计数
                return idx + 1
        except Exception:
            continue
    return None


def normalize(values: List[Optional[float]]) -> List[float]:
    """
    保留函数占位但不再使用（为兼容旧接口）。
    新逻辑：不做跨模型归一化，仅按维度比例尺映射到 0–1 半径。
    """
    valid = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not valid:
        return [0.0 for _ in values]
    # 直接返回原值（调用方应提供 0–1 范围或自行按比例尺映射）
    out: List[float] = []
    for v in values:
        try:
            out.append(float(v) if v is not None and not math.isnan(float(v)) else 0.0)
        except Exception:
            out.append(0.0)
    return out


def read_speed_benchmark(csv_path: Path) -> Dict[str, float]:
    """
    读取速度基准 CSV（result/benchmark_speed_subset.csv），返回字典：`模型代号 -> FPS`。
    CSV 表头约定：
    - model_code, model_label, weights, preprocess_ms_per_img, inference_ms_per_img,
      postprocess_ms_per_img, avg_total_ms_per_img, fps
    """
    m: Dict[str, float] = {}
    if not csv_path.exists():
        return m
    try:
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
    except Exception:
        return m
    return m


def main() -> None:
    """
    主入口：读取 7 个模型在 6 个维度的指标并绘制雷达图。
    """
    setup_cn_font()

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=str(ROOT / "result" / "radar_chart_final.png"), help="输出图片路径")
    parser.add_argument("--dpi", type=int, default=140, help="保存图像的 DPI")
    parser.add_argument("--summary", type=str, default=str(ROOT / "result" / "多模型结果汇总.csv"), help="汇总表路径（优先读取）")
    parser.add_argument("--show", action="store_true", help="保存后同时在本机弹窗预览")
    parser.add_argument("--annotate-values", action="store_true", help="是否在各维度绘制数据真值与引线（默认不显示）")
    args = parser.parse_args()

    # 模型注册表（严格 7 个项，与文档一致）
    # 每项包括：展示名称、结果 CSV 路径（测试集）、训练过程 results.csv 路径
    registry: List[Dict[str, str]] = [
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

    # 优先从“多模型结果汇总.csv”读取所有维度；若缺失则回退到原有散点来源
    labels = [r["label"] for r in registry]
    codes = [r["code"] for r in registry]
    summary_path = Path(args.summary)
    map50_raw: List[Optional[float]] = []
    map95_raw: List[Optional[float]] = []
    recall_raw: List[Optional[float]] = []
    precision_raw: List[Optional[float]] = []
    fps_raw: List[Optional[float]] = []
    conv_eff_raw: List[Optional[float]] = []

    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = {row.get("model_code"): row for row in reader}
            for code in codes:
                row = rows.get(code, {})
                def _getf(name: str) -> Optional[float]:
                    v = row.get(name)
                    try:
                        return float(v) if v not in (None, "") else None
                    except Exception:
                        return None
                map50_raw.append(_getf("mAP50"))
                map95_raw.append(_getf("mAP95"))
                recall_raw.append(_getf("Recall"))
                precision_raw.append(_getf("Precision"))
                fps_raw.append(_getf("FPS"))
                # 若未提供 conv_eff，则回退由 epochs_to_0.6 计算
                conv_eff = _getf("conv_eff")
                if conv_eff is None:
                    ep_v = row.get("epochs_to_0.6")
                    try:
                        ep_i = int(ep_v) if ep_v not in (None, "") else 0
                        conv_eff = (100.0 / float(ep_i)) if ep_i > 0 else None
                    except Exception:
                        conv_eff = None
                conv_eff_raw.append(conv_eff)
        except Exception:
            summary_path = Path("")  # 强制回退

    if not summary_path.exists():
        eval_paths = [Path(r["eval_csv"]) for r in registry]
        train_paths = [Path(r["train_csv"]) for r in registry]
        map50_raw = [read_single_value_csv(p, "metrics/mAP50(B)") for p in eval_paths]
        map95_raw = [read_single_value_csv(p, "metrics/mAP50-95(B)") for p in eval_paths]
        recall_raw = [read_single_value_csv(p, "metrics/recall(B)") for p in eval_paths]
        precision_raw = [read_single_value_csv(p, "metrics/precision(B)") for p in eval_paths]
        speed_csv = ROOT / "result" / "benchmark_speed_subset.csv"
        speed_map = read_speed_benchmark(speed_csv)
        fps_raw = []
        for r in registry:
            code = r.get("code", "")
            v = speed_map.get(code)
            fps_raw.append(None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v))
        conv_eff_raw = []
        for tp in train_paths:
            series = read_training_series(tp, "metrics/mAP50(B)")
            ep = first_epoch_reach_threshold(series, 0.6)
            if ep is None or ep <= 0:
                conv_eff_raw.append(None)
            else:
                conv_eff_raw.append(100.0 / float(ep))

    # === 特殊规则：微调模型 M6 不参与收敛速度对比 ===
    # 将其收敛效率置为 None，避免参与范围计算与线条绘制中的误导
    for i, code in enumerate(codes):
        if code == "M6":
            conv_eff_raw[i] = None

    # ============ 取消统一归一化：按维度“定制比例尺”映射到 0–1 半径 ============
    # 通用“漂亮边界”选择：为每个维度选取 [lower, upper]，提升散布与可读性
    def _nice_bounds(values: List[Optional[float]], base: float, low_ratio: float, high_ratio: float) -> Tuple[float, float]:
        vs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
        if not vs:
            return 0.0, 1.0
        vmin = min(vs)
        vmax = max(vs)
        lower = math.floor((vmin * (1.0 - low_ratio)) / base) * base
        upper = math.ceil((vmax * (1.0 + high_ratio)) / base) * base
        if upper <= lower:
            upper = lower + base
        return lower, upper

    # 为四个精度维度设置边界（不从 0 开始），按 1% 精度刻度并留 2% 边距
    map50_lo, map50_up = _nice_bounds(map50_raw, base=0.01, low_ratio=0.02, high_ratio=0.02)
    map95_lo, map95_up = _nice_bounds(map95_raw, base=0.01, low_ratio=0.02, high_ratio=0.02)
    recall_lo, recall_up = _nice_bounds(recall_raw, base=0.01, low_ratio=0.02, high_ratio=0.02)
    precision_lo, precision_up = _nice_bounds(precision_raw, base=0.01, low_ratio=0.02, high_ratio=0.02)

    # FPS：5% 边距，5 为刻度单位；同时给下界，避免零起点过于拥挤
    fps_lo, fps_up = _nice_bounds(fps_raw, base=5.0, low_ratio=0.05, high_ratio=0.05)

    # Convergence：对收敛效率做 log1p 压缩，避免极端值主导；刻度单位 0.1
    conv_log_raw: List[Optional[float]] = []
    for v in conv_eff_raw:
        try:
            conv_log_raw.append(math.log1p(float(v)) if v is not None and not math.isnan(float(v)) else None)
        except Exception:
            conv_log_raw.append(None)
    conv_lo, conv_up = _nice_bounds(conv_log_raw, base=0.1, low_ratio=0.03, high_ratio=0.03)

    def _scale_range(v: Optional[float], lo: float, up: float) -> float:
        try:
            val = float(v) if v is not None and not math.isnan(float(v)) else None
            if val is None:
                return 0.0
            return max(0.0, min(1.0, (val - lo) / (up - lo)))
        except Exception:
            return 0.0

    map50_n = [_scale_range(v, map50_lo, map50_up) for v in map50_raw]
    map95_n = [_scale_range(v, map95_lo, map95_up) for v in map95_raw]
    recall_n = [_scale_range(v, recall_lo, recall_up) for v in recall_raw]
    precision_n = [_scale_range(v, precision_lo, precision_up) for v in precision_raw]
    fps_n = [_scale_range(v, fps_lo, fps_up) for v in fps_raw]
    conv_n = [_scale_range((math.log1p(v) if v is not None else None), conv_lo, conv_up) for v in conv_eff_raw]

    # 雷达图维度与角度
    # 在轴标签中标注各维度的范围（按数据结果设计），提升可读性
    def _rng(a: float, b: float, nd: int = 2) -> str:
        return f"{a:.{nd}f}–{b:.{nd}f}"

    metrics_names = [
        f"mAP50（平均精度@50%）[{_rng(map50_lo, map50_up)}]",
        f"mAP50‑95（平均精度@[50,95]%）[{_rng(map95_lo, map95_up)}]",
        f"Recall（召回率）[{_rng(recall_lo, recall_up)}]",
        f"Precision（精确率）[{_rng(precision_lo, precision_up)}]",
        f"FPS（帧率）[{int(fps_lo)}–{int(fps_up)}]",
        f"Convergence（收敛效率，log1p）[{_rng(min([v for v in conv_eff_raw if v is not None], default=0.0), max([v for v in conv_eff_raw if v is not None], default=1.0), nd=0)}]",
    ]
    data_by_model: List[List[float]] = []
    for i in range(len(registry)):
        data_by_model.append([
            map50_n[i],
            map95_n[i],
            recall_n[i],
            precision_n[i],
            fps_n[i],
            conv_n[i],
        ])

    # 每条线需要闭合
    def _closed(vals: List[float]) -> List[float]:
        return vals + [vals[0]]

    # 角度分配
    N = len(metrics_names)
    angles = [n / float(N) * 2.0 * math.pi for n in range(N)]
    angles += angles[:1]  # 闭合

    # 绘图
    # 放大画布，减少文字堆叠风险
    fig, ax = plt.subplots(figsize=(13.5, 11.0), subplot_kw=dict(polar=True))

    # 网格与角度标签
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names)
    ax.tick_params(axis="x", labelsize=10, pad=8)
    # 只保留网格，不显示半径刻度标签（删除 0.2~1.0 的遗留刻度文字）
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.12)
    ax.grid(True, linestyle="--", alpha=0.3)

    # 绘制每个模型
    for idx, model_vals in enumerate(data_by_model):
        color = COLOR_POOL[idx % len(COLOR_POOL)]
        ax.plot(angles, _closed(model_vals), color=color, linewidth=2.0, linestyle="-", label=labels[idx])
        ax.fill(angles, _closed(model_vals), color=color, alpha=0.12)

    # 在各维度上为每个模型标注“数据真值”（可开关），并做抖动避免堆叠
    def _fmt_val(dim: int, v: Optional[float]) -> str:
        if v is None or (isinstance(v, float) and math.isnan(float(v))):
            return "N/A"
        if dim in (0, 1, 2, 3):
            return f"{float(v):.4f}"
        if dim == 4:
            return f"{float(v):.0f}"
        if dim == 5:
            return f"{float(v):.2f}"
        return str(v)

    if args.annotate_values:
        raw_matrix: List[List[Optional[float]]] = [
            map50_raw,
            map95_raw,
            recall_raw,
            precision_raw,
            fps_raw,
            conv_eff_raw,
        ]

        for j in range(len(metrics_names)):
            # 收集该维度所有模型的半径与索引，并按半径排序以分配抖动偏移
            items = []
            for i, vals in enumerate(data_by_model):
                items.append((vals[j], i))
            items.sort(key=lambda x: x[0])
            n = len(items)
            # 引线与标签参数：增大抖动与标签半径，降低堆叠概率
            base_dt = 0.035  # 角度抖动（弧度）
            base_rt = 0.09   # 半径抖动（用于端点，避免完全重合）
            label_r = 1.18   # 标签半径（图外）
            for k, (r, i) in enumerate(items):
                color = COLOR_POOL[i % len(COLOR_POOL)]
                ang = angles[j]
                # 端点抖动：在半径与角度上根据排序位置分散
                offset_index = k - (n - 1) / 2.0
                ang_txt = ang + base_dt * offset_index
                r_end = max(0.04, min(0.98, r + base_rt * offset_index))
                txt = _fmt_val(j, raw_matrix[j][i])

                # 引线：从数据点（ang, r）到图外标签位置（ang_txt, label_r）
                ax.annotate(
                    txt,
                    xy=(ang, r),
                    xytext=(ang_txt, label_r),
                    textcoords='data',
                    arrowprops=dict(
                        arrowstyle='-',
                        color=color,
                        lw=1.2,
                        linestyle='--',
                        shrinkA=0,
                        shrinkB=0,
                        connectionstyle='arc3,rad=0.0',
                    ),
                    ha='center',
                    va='center',
                    fontsize=8,
                    bbox=dict(facecolor="white", edgecolor=color, alpha=0.65, boxstyle="round,pad=0.28"),
                )

    # 图例与注释
    # 添加说明：若所有 FPS 缺失则统一说明；并标注“M6不计收敛速度”
    fps_missing = all((v is None or (isinstance(v, float) and math.isnan(v))) for v in fps_raw)
    legend_title = ("模型（FPS缺失记0；M6不计收敛速度）" if fps_missing else "模型图例")
    # 图例置底部（基于 figure 而非 axes），多列排布避免遮挡
    handles, labels_used = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles, labels_used,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        title=legend_title,
        frameon=True,
    )
    leg.get_frame().set_alpha(0.75)

    # 标题
    plt.title("YOLO-Fusion-UA-Lite 最终综合评估雷达图（定制比例尺）", pad=20)

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 调整画布留白，确保左侧标签不出界、底部图例不遮挡
    plt.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.28)
    plt.savefig(out_path.as_posix(), dpi=args.dpi, bbox_inches="tight", pad_inches=0.08)
    print(
        "\n".join([
            f"已保存雷达图：{out_path.as_posix()}",
            f"比例尺：mAP50[{map50_lo:.2f}–{map50_up:.2f}], mAP50-95[{map95_lo:.2f}–{map95_up:.2f}],",
            f"         Recall[{recall_lo:.2f}–{recall_up:.2f}], Precision[{precision_lo:.2f}–{precision_up:.2f}],",
            f"         FPS[{int(fps_lo)}–{int(fps_up)}], Convergence(log1p)[{conv_lo:.2f}–{conv_up:.2f}] (原值范围用于标签)"
        ])
    )

    # 可选预览：保存后在本机弹窗打开，同时显示 Matplotlib 窗口（若环境支持）
    if args.show:
        # 尝试用默认查看器打开；若失败则用浏览器打开 file:// URL；最后尝试 Matplotlib 窗口
        opened = False
        try:
            if os.name == "nt":
                os.startfile(out_path.as_posix())
                opened = True
        except Exception:
            opened = False
        if not opened:
            try:
                webbrowser.open(out_path.as_uri(), new=2)
                opened = True
            except Exception:
                opened = False
        try:
            plt.show()  # 阻塞显示，避免窗口瞬间关闭
        except Exception:
            pass


if __name__ == "__main__":
    main()
