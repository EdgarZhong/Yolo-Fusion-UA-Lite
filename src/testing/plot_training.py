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
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except Exception:
    WATCHDOG_AVAILABLE = False

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager

# ===================== 配置区域（请按需修改） =====================
# 在 models/ 下的训练输出目录（含 results.csv）
MODEL_DIRS: List[str] = [
    "formal/fusion-attention/dualbackbone-fusionattention-obb",
    "formal/dualbackbone-easy-obb-formal6",
    "formal/dualbackbone-FA-Concat-obb",
    "formal/FA-Concat-FPN-PAN-neck",
    "formal/CM-FA-Concat-FPN-PAN-neck",
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


def _collect_series() -> tuple[Dict[str, Dict[str, List[float]]], Dict[str, List[float]], float]:
    """
    读取各模型 results.csv，返回：
    - series_by_model: 每模型的列数据字典
    - x_by_model: 每模型的 X 轴（epoch 或行号）
    - global_max_epoch: 全局最大 epoch
    """
    series_by_model: Dict[str, Dict[str, List[float]]] = {}
    x_by_model: Dict[str, List[float]] = {}
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
            except Exception:
                pass

    return series_by_model, x_by_model, global_max_epoch


def main() -> None:
    """
    主入口：加载各模型的 results.csv 并绘制折线图。
    """
    _setup_cn_font()

    # 命令行参数：支持 --watch 持续刷新
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="使用 watchdog 事件驱动监控 CSV 变化并自动刷新绘图")
    parser.add_argument("--debounce-ms", type=int, default=800, help="文件变动防抖时间窗口（毫秒），默认 800ms")
    parser.add_argument("--tick-ms", type=int, default=20000, help="UI 定时器周期（毫秒），默认 200ms")
    parser.add_argument("--no-window", action="store_true", help="watch 模式禁用交互窗口，改为静态图片刷新并保存到 result/")
    args = parser.parse_args()

    # 初次绘制
    series_by_model, x_by_model, global_max_epoch = _collect_series()
    if not series_by_model:
        print("未成功加载任何 results.csv，结束")
        return

    # 2. 准备绘图
    cols_layout = 2
    rows_layout = math.ceil(len(PLOT_KEYS) / cols_layout)
    
    # 适当增加高度以容纳更多行
    fig, axes = plt.subplots(rows_layout, cols_layout, figsize=(15, 3.5 * rows_layout), squeeze=False)
    # 非 watch 模式或需要交互窗口时才启用交互；watch+no-window 下不创建交互事件循环
    if not args.watch or (args.watch and not args.no_window):
        plt.ion()

    legend_handles = []
    legend_labels = []
    seen_labels = set()

    # 3. 逐个指标绘图（封装为函数，便于刷新）
    def draw_all(series_by_model: Dict[str, Dict[str, List[float]]], x_by_model: Dict[str, List[float]], global_max_epoch: float):
        # 清空旧图线
        for ax_row in axes:
            for ax in ax_row:
                ax.clear()

        nonlocal legend_handles, legend_labels, seen_labels
        legend_handles = []
        legend_labels = []
        seen_labels = set()

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
                    lines = ax.get_lines()
                    if lines:
                        line = lines[-1]
                        legend_handles.append(line)
                        legend_labels.append(rel_dir)
                        seen_labels.add(rel_dir)

            # 统一横轴范围
            if global_max_epoch > 0:
                ax.set_xlim(0, global_max_epoch)

        # 重新添加图例与布局
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.92),
                ncol=min(3, len(legend_labels)),
                frameon=True,
                fontsize=10,
                borderaxespad=0.0,
            )
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.subplots_adjust(hspace=0.4, wspace=0.2)
        fig.canvas.draw()
        plt.pause(0.01)

    # 4. 隐藏多余的子图位置
    total_subplots = rows_layout * cols_layout
    for j in range(len(PLOT_KEYS), total_subplots):
        r = j // cols_layout
        c = j % cols_layout
        axes[r][c].axis("off")

    # ================= 核心修改区域 =================
    
    # 首次绘制
    draw_all(series_by_model, x_by_model, global_max_epoch)
    # 仅在需要交互窗口时做非阻塞展示；静态模式无需弹窗
    if not args.watch or (args.watch and not args.no_window):
        plt.show(block=False)

    # 6. 调整布局
    # rect=[left, bottom, right, top]
    # top=0.90 表示所有子图只能画在画布高度的 0% 到 90% 之间
    # 留出顶部的 10% (0.90~1.00) 给图例，避免遮挡
    # 若启用监控，使用 watchdog 事件驱动 + 防抖 + 异步刷新
    if args.watch:
        if not WATCHDOG_AVAILABLE:
            print("错误：未检测到 watchdog 库。请先安装：pip install watchdog")
            return

        debounce_ms = max(100, int(args.debounce_ms))
        tick_ms = args.tick_ms

        class _DebounceAggregator:
            """文件变动防抖聚合器：记录各目录最近一次事件时间戳，超过窗口才认为就绪"""
            def __init__(self):
                self._lock = threading.Lock()
                self._last_ts: Dict[str, float] = {}

            def mark_changed(self, rel_dir: str) -> None:
                now = time.time()
                with self._lock:
                    self._last_ts[rel_dir] = now

            def pop_ready(self, window_ms: int) -> List[str]:
                now = time.time()
                ready: List[str] = []
                with self._lock:
                    for d, ts in list(self._last_ts.items()):
                        if (now - ts) * 1000.0 >= window_ms:
                            ready.append(d)
                            del self._last_ts[d]
                return ready

        aggregator = _DebounceAggregator()

        class _ResultsCsvHandler(FileSystemEventHandler):
            """仅监听 models/<rel_dir>/results.csv 的创建/修改事件，并触发聚合器标记"""
            def __init__(self, root: Path, rel_dirs: List[str], on_change):
                super().__init__()
                self.root = root
                self.base = (self.root / "models").resolve()
                self.rel_set = set(rel_dirs)
                self.on_change = on_change

            def _match_rel(self, src_path: str) -> Optional[str]:
                try:
                    p = Path(src_path).resolve()
                    if p.name != "results.csv":
                        return None
                    rel = p.parent.resolve().relative_to(self.base)
                    rel_str = str(rel).replace("\\", "/")
                    return rel_str if rel_str in self.rel_set else None
                except Exception:
                    return None

            def on_created(self, event):
                if getattr(event, "is_directory", False):
                    return
                rel_dir = self._match_rel(getattr(event, "src_path", ""))
                if rel_dir:
                    aggregator.mark_changed(rel_dir)

            def on_modified(self, event):
                if getattr(event, "is_directory", False):
                    return
                rel_dir = self._match_rel(getattr(event, "src_path", ""))
                if rel_dir:
                    aggregator.mark_changed(rel_dir)

        # 创建并启动文件系统观察者
        observer = Observer()
        handler = _ResultsCsvHandler(ROOT, MODEL_DIRS, aggregator.mark_changed)
        watched_any = False
        for rel_dir in MODEL_DIRS:
            watch_dir = (ROOT / "models" / rel_dir)
            if watch_dir.exists():
                observer.schedule(handler, str(watch_dir), recursive=False)
                watched_any = True
            else:
                print(f"警告：监控目录不存在，跳过 {watch_dir.as_posix()}")

        if not watched_any:
            print("未找到可监控的目录，结束")
            return

        observer.start()

        # 解析与刷新采用异步：在后台线程读取 CSV，前台 UI 线程仅负责绘制
        executor = ThreadPoolExecutor(max_workers=2)
        loading_future = None

        def _on_close(_event):
            try:
                observer.stop()
                observer.join(timeout=2.0)
            except Exception:
                pass
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        if not args.no_window:
            fig.canvas.mpl_connect("close_event", _on_close)

        def _tick():
            nonlocal loading_future, series_by_model, x_by_model, global_max_epoch
            # 窗口已关闭则停止观察者与定时器
            if not plt.fignum_exists(fig.number):
                try:
                    observer.stop()
                    observer.join(timeout=2.0)
                except Exception:
                    pass
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                return

            # 事件防抖后就绪，触发一次异步解析
            ready_dirs = aggregator.pop_ready(debounce_ms)
            if ready_dirs and loading_future is None:
                try:
                    loading_future = executor.submit(_collect_series)
                except Exception as e:
                    print(f"提交解析任务失败：{e}")
                    loading_future = None

            # 若后台解析已完成，则刷新绘图
            if loading_future is not None and loading_future.done():
                try:
                    series_by_model, x_by_model, global_max_epoch = loading_future.result()
                    draw_all(series_by_model, x_by_model, global_max_epoch)
                except Exception as e:
                    print(f"刷新绘图失败：{e}")
                finally:
                    loading_future = None

            try:
                fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception:
                pass

        # 两种模式：
        # 1) 交互窗口 + UI 定时器 + watchdog 事件驱动刷新
        # 2) 静态图片刷新（无窗口）：仅在防抖就绪时异步解析并保存 PNG
        if not args.no_window:
            ui_timer = fig.canvas.new_timer(interval=tick_ms)
            ui_timer.add_callback(_tick)
            ui_timer.start()
        else:
            # 静态刷新：事件驱动 + 防抖 + 异步解析 + 保存 PNG 文件
            out_png = ROOT / "result" / "plot_training_watch.png"

            # 定义静态绘制函数（强调最后一个模型：更粗线条与更大点）
            def render_static():
                try:
                    s_by_model, x_model, g_max = _collect_series()
                    # 创建独立画布，避免与交互画布共享状态
                    rows = rows_layout
                    cols = cols_layout
                    fig2, axes2 = plt.subplots(rows, cols, figsize=(15, 3.5 * rows), squeeze=False)
                    _setup_cn_font()

                    last_key = MODEL_DIRS[-1] if MODEL_DIRS else None
                    for idx_key, key in enumerate(PLOT_KEYS):
                        r = idx_key // cols
                        c = idx_key % cols
                        ax = axes2[r][c]
                        ax.set_title(key, fontsize=11, fontweight='bold')
                        ax.set_xlabel("Epoch", fontsize=9)
                        ax.set_ylabel("Value", fontsize=9)

                        legend_handles2 = []
                        legend_labels2 = []

                        for idx_model, rel_dir in enumerate(MODEL_DIRS):
                            if rel_dir not in s_by_model:
                                continue
                            series = s_by_model[rel_dir]
                            xs = x_model.get(rel_dir, [])
                            ys = series.get(key, [])
                            if not ys:
                                continue

                            color = _get_model_color(rel_dir, idx_model)
                            # 区分最后一条配置的模型：加粗线条与更大的点
                            lw = 2.6 if rel_dir == last_key else 1.0
                            ms = 6 if rel_dir == last_key else 3
                            ls = "-" if rel_dir == last_key else "--"
                            marker = "o" if rel_dir == last_key else "."

                            # 直接绘制并收集图例项
                            xs_clean = []
                            ys_clean = []
                            for xi, yi in zip(xs, ys):
                                try:
                                    v = float(yi)
                                    if math.isnan(v):
                                        continue
                                except Exception:
                                    continue
                                xs_clean.append(float(xi))
                                ys_clean.append(v)
                            if xs_clean and ys_clean:
                                line = axes2[r][c].plot(xs_clean, ys_clean, color=color, linewidth=lw, linestyle=ls, marker=marker, markersize=ms, label=rel_dir)[0]
                                legend_handles2.append(line)
                                legend_labels2.append(rel_dir)
                            axes2[r][c].grid(True, linestyle="--", alpha=0.3)
                            if g_max > 0:
                                axes2[r][c].set_xlim(0, g_max)

                        if legend_handles2:
                            fig2.legend(legend_handles2, legend_labels2, loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=min(3, len(legend_labels2)), frameon=True, fontsize=10, borderaxespad=0.0)

                    # 隐藏多余子图
                    total_sub = rows * cols
                    for j in range(len(PLOT_KEYS), total_sub):
                        rr = j // cols
                        cc = j % cols
                        axes2[rr][cc].axis("off")

                    plt.tight_layout(rect=[0, 0, 1, 0.90])
                    plt.subplots_adjust(hspace=0.4, wspace=0.2)
                    fig2.canvas.draw()
                    fig2.savefig(out_png, dpi=150)
                except Exception as e:
                    print(f"静态绘图保存失败：{e}")
                finally:
                    try:
                        plt.close(fig2)
                    except Exception:
                        pass

            # 定义事件驱动的防抖调度器：每次文件变动重置定时器，到期后提交异步渲染
            class DebounceScheduler:
                def __init__(self, window_ms: int):
                    self.window = max(100, int(window_ms)) / 1000.0
                    self._timer: Optional[threading.Timer] = None
                    self._lock = threading.Lock()
                    self._running_future = None

                def schedule(self):
                    with self._lock:
                        if self._timer is not None:
                            try:
                                self._timer.cancel()
                            except Exception:
                                pass
                        self._timer = threading.Timer(self.window, self._fire)
                        self._timer.daemon = True
                        self._timer.start()

                def _fire(self):
                    with self._lock:
                        if self._running_future is None or self._running_future.done():
                            try:
                                self._running_future = executor.submit(render_static)
                            except Exception as e:
                                print(f"提交静态绘图任务失败：{e}")
                        # 若已有任务在跑，直接忽略，等待其完成

                def shutdown(self):
                    with self._lock:
                        if self._timer is not None:
                            try:
                                self._timer.cancel()
                            except Exception:
                                pass
                        try:
                            if self._running_future is not None:
                                _ = self._running_future.result(timeout=10)
                        except Exception:
                            pass

            scheduler = DebounceScheduler(debounce_ms)

            # 事件处理器触发调度器
            def _mark_and_schedule(rel_dir: str):
                try:
                    aggregator.mark_changed(rel_dir)
                    scheduler.schedule()
                except Exception:
                    pass

            # 替换 handler 的标记函数为调度器触发
            handler = _ResultsCsvHandler(ROOT, MODEL_DIRS, _mark_and_schedule)
            # 重新绑定观察者（前面已 schedule 原 handler 的场景，这里确保 handler 一致）
            try:
                observer.unschedule_all()
            except Exception:
                pass
            watched_any2 = False
            for rel_dir in MODEL_DIRS:
                watch_dir = (ROOT / "models" / rel_dir)
                if watch_dir.exists():
                    observer.schedule(handler, str(watch_dir), recursive=False)
                    watched_any2 = True
            if not watched_any2:
                print("未找到可监控的目录，结束")
                try:
                    observer.stop(); observer.join(timeout=2.0)
                except Exception:
                    pass
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                return

            # 启动一次初始绘制
            scheduler.schedule()

            # headless 主线程保持运行至 Ctrl+C，完全事件驱动无 GUI 阻塞
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("用户中断，退出静态监控模式")
                try:
                    scheduler.shutdown()
                except Exception:
                    pass
                try:
                    observer.stop(); observer.join(timeout=2.0)
                except Exception:
                    pass
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass

        # 交互窗口：阻塞式进入 GUI 事件循环，由后端负责事件派发
        if not args.no_window:
            try:
                # 交互模式关闭，确保 show 阻塞而不瞬退
                plt.ioff()
                plt.show()
            except KeyboardInterrupt:
                print("用户中断，退出监控模式")
                _on_close(None)

    else:
        print("绘图完成，未启用监控。若需自动刷新，请使用 --watch 参数。")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
