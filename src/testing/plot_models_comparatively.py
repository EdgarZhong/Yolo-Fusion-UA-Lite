"""
此脚本暂时存在bug和数据来源不足，未启用
多模型对比可视化脚本：读取 `result/` 下指定的多个测试结果 JSON 文件，绘制两张图：
1) 大图（网格）：每个模型一行，左侧为总体指标柱状图（precision(B)/recall(B)/mAP50(B)/mAP50-95(B)），右侧为 5x5 混淆矩阵；
2) 雷达图：多模型综合维度对比（mAP50、mAP50-95、Recall、FPS、Params、Small Object AP）。

使用约束：
- 顶部定义 `RESULT_FILES` 为文件名数组（不传命令行参数）；脚本在 `result/` 目录下按文件名查找。
- 若某些维度（如 FPS/Params/Small Object AP）在 JSON 中不可用，可在 `MODEL_INFO` 补充；
- 若无法获取混淆矩阵，则在右侧以“数据不可用”占位图展示。

使用：
- `python src/testing/plot_models_comparatively.py`
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager

# ===================== 配置区域（请按需修改） =====================
# 在 result/ 目录下要对比的结果文件名（包含 .json 扩展名）
RESULT_FILES: List[str] = [
    "baseline-100epoch.json",
    "fusionattention-only-120epoch.json",
]

# 额外模型信息（可选）：当 JSON 中缺失某些维度时使用此处的补充值
# 键为 JSON 文件名（与 RESULT_FILES 元素一致）；值为可选维度字典
MODEL_INFO: Dict[str, Dict[str, float]] = {
    # 示例："baseline-100epoch.json": {"fps": 55.0, "params": 8.2e6, "small_ap": 0.31}
}

# 固定中文颜色与模型颜色映射（同一模型在不同图表中颜色保持一致）
MODEL_COLORS: Dict[str, str] = {
    "baseline-100epoch.json": "#4e79a7",  # 蓝色
    "fusionattention-only-120epoch.json": "#e15759",  # 红色
}

# 仓库根路径与结果目录
ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "result"


def _setup_cn_font() -> str:
    """
    设置 Matplotlib 中文字体，避免中文乱码。
    返回：实际使用的字体名称（若未找到则返回空字符串）。
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


def _load_json(json_path: Path) -> dict:
    """
    读取指定评估结果 JSON。
    返回：包含总体指标与 `classes/names` 的字典。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_overall_metrics(d: dict) -> Tuple[List[str], List[float]]:
    """
    提取总体指标四项：precision(B), recall(B), mAP50(B), mAP50-95(B)
    返回：(标签列表, 数值列表[四项])
    """
    keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    vals = [float(d.get(k, 0.0)) for k in keys]
    labels = ["precision(B)", "recall(B)", "mAP50(B)", "mAP50-95(B)"]
    return labels, vals


def _infer_fps_params_smallap(d: dict, extra: Dict[str, float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    推断或补充 FPS、参数量（Params）、小目标 AP：
    - FPS：若 JSON 中存在 `speed` 字段（毫秒/图），则用 `1000 / (preprocess+inference+postprocess)` 近似；
      若无则使用 `extra` 中的 `fps`；
    - Params：JSON 通常无此字段，使用 `extra` 中的 `params`；
    - small_ap：若 JSON 中无小目标 AP，使用 `extra` 中的 `small_ap`（可为空）。
    返回：(fps, params, small_ap)，均可能为 None。
    """
    fps = None
    params = None
    small_ap = None
    speed = d.get("speed", None)
    if isinstance(speed, dict):
        ms = float(speed.get("preprocess", 0.0)) + float(speed.get("inference", 0.0)) + float(speed.get("postprocess", 0.0))
        if ms > 0:
            fps = 1000.0 / ms
    if extra:
        fps = fps if fps is not None else extra.get("fps")
        params = extra.get("params", params)
        small_ap = extra.get("small_ap", small_ap)
    return fps, params, small_ap


def _load_confusion_matrix(run_name: str, num_classes: int) -> Optional[np.ndarray]:
    """
    尝试加载混淆矩阵：优先读取 `result/<run_name>/confusion_matrix.npy`；若不存在返回 None。
    注意：由于评估脚本默认 `plots=False`，通常不会生成混淆矩阵文件；此处做“尽力而为”的加载。
    """
    cm_npy = RESULT_DIR / run_name / "confusion_matrix.npy"
    if cm_npy.exists():
        try:
            return np.load(cm_npy)
        except Exception:
            pass
    return None


def _plot_model_row(
    axes_row: Tuple[plt.Axes, plt.Axes],
    labels: List[str],
    values: List[float],
    cm: Optional[np.ndarray],
    class_names: List[str],
    model_title: str,
    color: str,
) -> None:
    """
    绘制一行：左侧总体指标柱状图；右侧混淆矩阵（或占位）。
    """
    ax_bar, ax_cm = axes_row
    bars = ax_bar.bar(labels, values, color=color)
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_title(f"{model_title} 总体指标")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.3)
    for b, v in zip(bars, values):
        ax_bar.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.02, f"{v:.3f}", ha="center", va="bottom")

    if cm is not None and cm.size:
        # 若类别数超过 5，仅展示前 5x5 的矩阵以保持版面一致
        cm_plot = cm[:5, :5] if cm.shape[0] >= 5 and cm.shape[1] >= 5 else cm
        im = ax_cm.imshow(cm_plot, cmap="Blues")
        ax_cm.set_title(f"{model_title} 混淆矩阵")
        ax_cm.set_xticks(range(len(class_names)))
        ax_cm.set_yticks(range(len(class_names)))
        ax_cm.set_xticklabels(class_names, rotation=45, ha="right")
        ax_cm.set_yticklabels(class_names)
        plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    else:
        ax_cm.axis("off")
        ax_cm.text(0.5, 0.5, "混淆矩阵数据不可用", ha="center", va="center")


def _plot_radar(models: List[str], metrics_map: Dict[str, Dict[str, Optional[float]]]) -> None:
    """
    绘制雷达图：维度按要求固定为 6 个。
    `metrics_map[file]` 需包含：map50, map5095, recall, fps, params, small_ap
    """
    dims = [
        "mAP50",
        "mAP50-95",
        "Recall",
        "FPS",
        "Params",
        "Small AP",
    ]

    # 归一化处理：除 mAP/Recall 外，其余量纲可能差异大，这里做简单线性缩放以便可视化
    def _normalize(values: List[Optional[float]]) -> List[float]:
        arr = np.array([0.0 if v is None else float(v) for v in values], dtype=float)
        max_v = arr.max() if arr.size else 1.0
        min_v = arr.min() if arr.size else 0.0
        if max_v - min_v > 1e-6:
            arr = (arr - min_v) / (max_v - min_v)
        return arr.tolist()

    theta = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_title("多模型综合对比雷达图")
    ax.set_xticks(theta)
    ax.set_xticklabels(dims)
    ax.set_yticklabels([])

    for file in models:
        m = metrics_map[file]
        raw_vals = [m.get("map50"), m.get("map5095"), m.get("recall"), m.get("fps"), m.get("params"), m.get("small_ap")]
        # 对 mAP/Recall 做截断到 [0,1]
        raw_vals[:3] = [min(1.0, max(0.0, 0.0 if v is None else float(v))) for v in raw_vals[:3]]
        # 其余做归一化（在所有模型之间）
        # 收集所有模型对应维度的原始值，用于缩放
    
    # 计算每个维度的归一化值（除前三个维度）
    matrices = {d: [] for d in dims}
    for file in models:
        m = metrics_map[file]
        matrices["mAP50"].append(0.0 if m.get("map50") is None else float(m.get("map50")))
        matrices["mAP50-95"].append(0.0 if m.get("map5095") is None else float(m.get("map5095")))
        matrices["Recall"].append(0.0 if m.get("recall") is None else float(m.get("recall")))
        matrices["FPS"].append(0.0 if m.get("fps") is None else float(m.get("fps")))
        matrices["Params"].append(0.0 if m.get("params") is None else float(m.get("params")))
        matrices["Small AP"].append(0.0 if m.get("small_ap") is None else float(m.get("small_ap")))

    def _norm_dim(values: List[float]) -> List[float]:
        arr = np.array(values, dtype=float)
        max_v = arr.max() if arr.size else 1.0
        min_v = arr.min() if arr.size else 0.0
        if max_v - min_v > 1e-6:
            arr = (arr - min_v) / (max_v - min_v)
        return arr.tolist()

    # 将所有模型的值转换为 0-1 范围
    norm_map: Dict[str, List[float]] = {}
    for dim in dims:
        norm_map[dim] = _norm_dim(matrices[dim])

    # 逐模型绘制闭合曲线
    for idx, file in enumerate(models):
        color = MODEL_COLORS.get(file, f"C{idx}")
        values = [
            norm_map["mAP50"][idx],
            norm_map["mAP50-95"][idx],
            norm_map["Recall"][idx],
            norm_map["FPS"][idx],
            norm_map["Params"][idx],
            norm_map["Small AP"][idx],
        ]
        values = np.array(values)
        ax.plot(theta, values, color=color, linewidth=2, label=file)
        ax.fill(theta, values, color=color, alpha=0.15)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    out_png = RESULT_DIR / "models_comparative_radar.png"
    plt.tight_layout()
    plt.savefig(out_png.as_posix(), dpi=200)
    print(f"已保存雷达图：{out_png.as_posix()}")


def main() -> None:
    """
    主入口：
    - 根据 RESULT_FILES 读取多个 JSON；
    - 绘制大图（每模型一行：总体柱状图 + 混淆矩阵）；
    - 绘制雷达图（多维度综合对比）。
    """
    _setup_cn_font()

    # 读取 JSON 并收集绘图所需信息
    loaded: List[Tuple[str, dict]] = []  # (文件名, 数据)
    for file in RESULT_FILES:
        jp = RESULT_DIR / file
        if not jp.exists():
            print(f"警告：未找到 {jp.name}，已跳过")
            continue
        try:
            d = _load_json(jp)
        except Exception as e:
            print(f"跳过 {jp.name}：读取失败 {e}")
            continue
        loaded.append((file, d))

    if not loaded:
        print("未成功加载任何结果 JSON，结束")
        return

    # 大图：每个模型一行，两列（1=总体柱状图；2=混淆矩阵）
    n = len(loaded)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 6 + 3 * n))
    if n == 1:
        axes = np.array([axes])  # 统一二维数组索引

    # 准备雷达图数据
    metrics_map: Dict[str, Dict[str, Optional[float]]] = {}

    for row_idx, (file, d) in enumerate(loaded):
        color = MODEL_COLORS.get(file, f"C{row_idx}")
        labels, values = _extract_overall_metrics(d)
        names_dict = d.get("names", {})
        class_names = [names_dict.get(i, str(i)) for i in range(min(5, len(names_dict)))] if isinstance(names_dict, dict) else [str(i) for i in range(5)]
        run_name = Path(file).stem
        cm = _load_confusion_matrix(run_name, num_classes=len(names_dict) if isinstance(names_dict, dict) else 5)
        _plot_model_row((axes[row_idx, 0], axes[row_idx, 1]), labels, values, cm, class_names, run_name, color)

        # 收集雷达图所需指标
        extra = MODEL_INFO.get(file, {})
        fps, params, small_ap = _infer_fps_params_smallap(d, extra)
        metrics_map[file] = {
            "map50": float(d.get("metrics/mAP50(B)", 0.0)),
            "map5095": float(d.get("metrics/mAP50-95(B)", 0.0)),
            "recall": float(d.get("metrics/recall(B)", 0.0)),
            "fps": fps,
            "params": params,
            "small_ap": small_ap,
        }

    # 保存大图
    out_grid = RESULT_DIR / "models_comparative_grid.png"
    plt.tight_layout()
    plt.savefig(out_grid.as_posix(), dpi=200)
    print(f"已保存多模型网格图：{out_grid.as_posix()}")

    # 绘制并保存雷达图
    _plot_radar([f for (f, _) in loaded], metrics_map)


if __name__ == "__main__":
    # 确保 Matplotlib 在非交互环境也能工作
    if "matplotlib" in sys.modules:
        import matplotlib

        matplotlib.use("Agg")
    main()
