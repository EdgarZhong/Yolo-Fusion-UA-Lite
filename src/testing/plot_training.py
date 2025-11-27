"""
多模型训练指标折线图绘制：在文件顶部定义要对比的模型目录数组，脚本到 `models/<目录>/results.csv`
读取训练/验证过程的关键指标，并在一张大图中分别绘制每个指标的折线图。

修改说明：
- 修复了图例遮挡子图的问题：通过调整 tight_layout 的 rect 参数为顶部预留空间。
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Dict, List, Optional
import math  # 确保 math 被导入

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager

# ===================== 配置区域（请按需修改） =====================
# 在 models/ 下的训练输出目录（含 results.csv）
MODEL_DIRS: List[str] = [
    "formal/fusion-attention/dualbackbone-fusionattention-obb",
    "formal/dualbackbone-easy-obb-formal6",
    "formal/dualbackbone-FA-Concat-obb"
    # 可追加其他目录，例如："formal/baseline/dualbackbone-obb"
]

# 同一模型在不同图表中颜色统一（键为目录路径字符串）
MODEL_COLORS: Dict[str, str] = {
    "formal/fusion-attention/dualbackbone-fusionattention-obb": "#e15759",  # 红色
    "formal/dualbackbone-easy-obb-formal6": "#4e79a7",  # 蓝色
}
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
_MODEL_COLOR_CACHE: Dict[str, str] = {}

def _get_model_color(model_key: str, idx: int) -> str:
    """
    返回统一的颜色：若在 MODEL_COLORS 中定义则使用；否则从 COLOR_POOL 自动分配并缓存。
    """
    if model_key in MODEL_COLORS:
        return MODEL_COLORS[model_key]
    if model_key in _MODEL_COLOR_CACHE:
        return _MODEL_COLOR_CACHE[model_key]
    color = COLOR_POOL[idx % len(COLOR_POOL)]
    _MODEL_COLOR_CACHE[model_key] = color
    return color

# 需要绘制的指标（按列名匹配）
PLOT_KEYS: List[str] = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "train/box_loss",
    "val/box_loss",
]

# 仓库根路径
ROOT = Path(__file__).resolve().parents[2]


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


def _read_results_csv(csv_path: Path) -> Dict[str, List[float]]:
    """
    读取 `results.csv` 并返回列名到数值列表的字典。
    若存在 `epoch` 列则返回；否则以行号作为 x 轴。
    """
    cols: Dict[str, List[float]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [fn.strip() if isinstance(fn, str) else fn for fn in reader.fieldnames]

        for row in reader:
            sanitized = {}
            for k, v in row.items():
                k2 = k.strip() if isinstance(k, str) else k
                v2 = v.strip() if isinstance(v, str) else v
                sanitized[k2] = v2
            for k2, v2 in sanitized.items():
                if k2 not in cols:
                    cols[k2] = []
                try:
                    cols[k2].append(float(v2))
                except Exception:
                    cols[k2].append(float("nan"))
    return cols


def _plot_series(ax: plt.Axes, x: List[float], y: List[float], label: str, color: str) -> None:
    """
    在给定坐标轴上绘制单条折线。
    """
    xs: List[float] = []
    ys: List[float] = []
    for xi, yi in zip(x, y):
        if yi is None:
            continue
        try:
            v = float(yi)
        except Exception:
            continue
        if math.isnan(v):
            continue
        xs.append(float(xi))
        ys.append(v)
    if not xs or not ys:
        return
    
    # 这里的 label 设为 None，因为我们在全局图例中添加，避免子图重复处理
    ax.plot(
        xs,
        ys,
        label=label, # 仍保留 label 以便 ax.get_lines() 获取
        color=color,
        linewidth=1.0,
        linestyle="--",
        marker=".",
        markersize=3,
    )
    ax.grid(True, linestyle="--", alpha=0.3)


def main() -> None:
    """
    主入口：加载各模型的 results.csv 并绘制折线图。
    """
    _setup_cn_font()

    # 1. 读取数据
    series_by_model: Dict[str, Dict[str, List[float]]] = {}
    x_by_model: Dict[str, List[float]] = {}
    
    # 预检：计算最大 Epoch，用于统一横轴
    global_max_epoch = 0.0

    for rel_dir in MODEL_DIRS:
        csv_path = ROOT / "models" / rel_dir / "results.csv"
        if not csv_path.exists():
            print(f"警告：未找到 {csv_path.as_posix()}，跳过")
            continue
        
        cols = _read_results_csv(csv_path)
        series_by_model[rel_dir] = cols
        
        # 确定 X 轴数据
        if "epoch" in cols and len(cols["epoch"]) > 0:
            x_by_model[rel_dir] = cols["epoch"]
        else:
            x_by_model[rel_dir] = list(range(len(next(iter(cols.values()), []))))
            
        # 更新全局最大 Epoch
        if x_by_model[rel_dir]:
            try:
                curr_max = float(x_by_model[rel_dir][-1])
                if curr_max > global_max_epoch:
                    global_max_epoch = curr_max
            except:
                pass

    if not series_by_model:
        print("未成功加载任何 results.csv，结束")
        return

    # 2. 准备绘图
    cols_layout = 2
    rows_layout = math.ceil(len(PLOT_KEYS) / cols_layout)
    
    # 适当增加高度以容纳更多行
    fig, axes = plt.subplots(rows_layout, cols_layout, figsize=(15, 3.5 * rows_layout), squeeze=False)

    legend_handles = []
    legend_labels = []
    seen_labels = set()

    # 3. 逐个指标绘图
    for idx, key in enumerate(PLOT_KEYS):
        r = idx // cols_layout
        c = idx % cols_layout
        ax = axes[r][c]
        
        ax.set_title(key, fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        
        # 遍历所有模型
        for idx_model, rel_dir in enumerate(MODEL_DIRS):
            if rel_dir not in series_by_model:
                continue
            
            series = series_by_model[rel_dir]
            color = _get_model_color(rel_dir, idx_model)
            xs = x_by_model[rel_dir]
            ys = series.get(key, [])
            
            if not ys:
                continue
                
            _plot_series(ax, xs, ys, label=rel_dir, color=color)

            # 收集图例信息（只收集一次）
            if rel_dir not in seen_labels:
                # 获取刚刚画的那条线
                lines = ax.get_lines()
                if lines:
                    # 查找对应颜色的线句柄
                    line = lines[-1] 
                    legend_handles.append(line)
                    legend_labels.append(rel_dir)
                    seen_labels.add(rel_dir)

        # 统一横轴范围
        if global_max_epoch > 0:
            ax.set_xlim(0, global_max_epoch)

    # 4. 隐藏多余的子图位置
    total_subplots = rows_layout * cols_layout
    for j in range(len(PLOT_KEYS), total_subplots):
        r = j // cols_layout
        c = j % cols_layout
        axes[r][c].axis("off")

    # ================= 核心修改区域 =================
    
    # 5. 添加全局图例
    # loc="lower center" + bbox_to_anchor=(0.5, 0.92) 
    # 意思是：图例框的“底部中心”锚定在整个图表高度的 92% 处
    # 这样图例会显示在 92% 以上的区域
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center", 
            bbox_to_anchor=(0.5, 0.92), 
            ncol=min(3, len(legend_labels)),
            frameon=True,
            fontsize=10,
            borderaxespad=0.
        )

    # 6. 调整布局
    # rect=[left, bottom, right, top]
    # top=0.90 表示所有子图只能画在画布高度的 0% 到 90% 之间
    # 留出顶部的 10% (0.90~1.00) 给图例，避免遮挡
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    # 增加子图之间的间距
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    
    # ===============================================

    plt.show()


if __name__ == "__main__":
    main()