"""
通用单模型测试结果展示脚本：自动遍历 `result/` 目录下的所有 `*.json` 测试结果文件，
为每个 JSON 绘制一张指标汇总图片并按文件名保存到同目录。

功能特性：
- 自动查找：无需命令行参数，脚本会自动扫描 `result/*.json`；
- 指标展示：绘制 2x3 子图（前 5 类的 precision/recall/AP50/AP 柱状图 + 1 张总体指标柱状图）；
- 命名规则：对文件 `result/<name>.json`，输出 `result/<name>-metrics.png`；

使用：
- `python src/testing/plot_single_model_metrics.py`
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager

# 仓库根路径与结果目录（固定扫描 result/ 下的所有 *.json）
ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "result"


def _setup_cn_font() -> str:
    """
    设置 Matplotlib 中文字体，避免中文乱码。

    策略：
    - 优先从常见中文字体中选择（按顺序尝试），如：Microsoft YaHei/SimHei/SimSun/NSimSun/KaiTi 等；
    - 通过 rcParams 设定全局字体族与 sans-serif 候选列表；
    - 禁用坐标轴负号乱码（axes.unicode_minus=False）。
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
    # 设置候选列表，让 Matplotlib 从中选择已安装的字体
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = candidates + list(rcParams.get("font.sans-serif", []))
    rcParams["axes.unicode_minus"] = False

    # 尝试检测并返回第一个可用字体名称（不强制要求存在）
    for name in candidates:
        try:
            # fallback_to_default=False 时，未找到会抛出到默认字体；我们仅用于判断是否存在
            path = font_manager.findfont(name, fallback_to_default=False)
            if isinstance(path, str) and len(path):
                return name
        except Exception:
            continue
    return ""


def load_metrics(json_path: Path) -> dict:
    """
    读取指定评估结果 JSON。
    参数：
    - json_path：结果 JSON 文件绝对路径
    返回：包含 `classes` 与总体指标键（如 `metrics/precision(B)` 等）的字典。
    """
    if not json_path.exists():
        raise FileNotFoundError(f"未找到评估结果文件：{json_path.as_posix()}，请先运行测试脚本生成")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def plot_metrics(metrics: dict, out_png: Path) -> None:
    """
    在单个画布上绘制 6 张子图（2x3）：
    - 5 个类别的指标柱状图：precision / recall / AP50 / AP
    - 1 个总体指标柱状图：precision(B) / recall(B) / mAP50(B) / mAP50-95(B)
    输出保存到 `out_png`
    """
    # 读取每类详细指标与总体指标
    classes = metrics.get("classes", [])
    names = metrics.get("names", {})
    overall_keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    overall_vals = [float(metrics.get(k, 0.0)) for k in overall_keys]

    # 确定绘制的类别顺序（按索引排序）
    classes_sorted = sorted(classes, key=lambda x: x.get("id", 0))
    # 若类别超过 5，仅取前 5 个以构成 6 子图（含总体）
    classes_sorted = classes_sorted[:5]

    # 统一颜色与指标键名
    bar_labels = ["precision", "recall", "ap50", "ap"]
    bar_colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    # 绘制每个类别的子图
    for i, cls_item in enumerate(classes_sorted):
        ax = axes[i]
        vals = [float(cls_item.get(k, 0.0)) for k in bar_labels]
        bars = ax.bar(bar_labels, vals, color=bar_colors)
        ax.set_ylim(0.0, 1.0)
        cname = cls_item.get("name", str(cls_item.get("id", i)))
        ax.set_title(f"类：{cname}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.02, f"{v:.3f}", ha="center", va="bottom")

    # 绘制总体指标子图（放在最后一个位置）
    ax_overall = axes[-1]
    bars = ax_overall.bar(["precision(B)", "recall(B)", "mAP50(B)", "mAP50-95(B)"], overall_vals, color=bar_colors)
    ax_overall.set_ylim(0.0, 1.0)
    ax_overall.set_title("总体指标")
    ax_overall.grid(axis="y", linestyle="--", alpha=0.3)
    for b, v in zip(bars, overall_vals):
        ax_overall.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.02, f"{v:.3f}", ha="center", va="bottom")

    # 若不足 5 类，其余子图隐藏
    for j in range(len(classes_sorted), 5):
        axes[j].axis("off")

    # 在图片顶部添加模型名称（取输出文件名的 stem）
    fig.suptitle(f"模型测试结果：{out_png.stem}", fontsize=12, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_png.as_posix(), dpi=200)
    plt.close(fig)
    print(f"已保存图表：{out_png.as_posix()}")


def main():
    """
    主入口：
    - 设置中文字体环境；
    - 自动遍历 `result/` 目录下的所有 `*.json` 文件；
    - 为每个文件生成同名的 `-metrics.png` 图片。
    """
    _setup_cn_font()

    if not RESULT_DIR.exists():
        print(f"结果目录不存在：{RESULT_DIR.as_posix()}，请先运行测试评估脚本生成 JSON 结果")
        return

    json_files = sorted(RESULT_DIR.glob("*.json"))
    if not json_files:
        print(f"未在 {RESULT_DIR.as_posix()} 发现任何 JSON 结果文件")
        return

    for jf in json_files:
        try:
            metrics = load_metrics(jf)
        except Exception as e:
            print(f"跳过文件 {jf.name}：读取失败 {e}")
            continue
        out_png = jf.with_name(f"{jf.stem}-metrics.png")
        plot_metrics(metrics, out_png)


if __name__ == "__main__":
    main()
