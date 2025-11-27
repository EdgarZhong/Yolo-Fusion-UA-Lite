"""
看门狗脚本：监控并自动重启 OBB 断点续训（resume_train.py）

功能与设计：
- 当训练进程异常退出（例如 CUDA 非法访问）时，按预设的重试次数与冷却时间自动重启。
- 默认启用 `CUDA_LAUNCH_BLOCKING=1`，便于在子进程内同步报错、快速定位问题。
- 只透传必要的三项覆盖参数（imgsz/batch/device），其余配置由检查点恢复，保持一致性。

使用示例：
- 基本用法（默认从上次正式训练目录恢复）：
  `python src/trainning/watchdog_resume.py`
- 自定义断点与覆盖：
  `python src/trainning/watchdog_resume.py --resume D:\Code\DeepLearning\YOLO-Fusion-UA-Lite\models\formal\fusion-attention\dualbackbone-fusionattention-obb\weights\last.pt --imgsz 832 --batch 12 --device 0`
- 控制重试：
  `python src/trainning/watchdog_resume.py --max-retries 10 --cooldown 60 --backoff 1.5`
"""

import argparse
import os
import sys
import time
from pathlib import Path
import subprocess


# 仓库根路径与本地 ultralytics 源码位置（用于默认路径与 CWD）
ROOT = Path(__file__).resolve().parents[2]


# def default_resume_path() -> Path:
#     """返回默认的 last.pt 路径，如果不存在则返回运行目录路径用于框架内部查找。"""
#     run_dir = ROOT / "models" / "formal" / "fusion-attention" / "dualbackbone-fusionattention-obb"
#     last_pt = run_dir / "weights" / "last.pt"
#     return last_pt if last_pt.exists() else run_dir


def build_command(args: argparse.Namespace) -> list[str]:
    """构建调用 resume_train.py 的命令参数列表。"""
    cmd = [sys.executable, str(ROOT / "src" / "trainning" / "resume_train.py")]
    if args.imgsz is not None:
        cmd += ["--imgsz", str(args.imgsz)]
    if args.batch is not None:
        cmd += ["--batch", str(args.batch)]
    if args.device is not None:
        cmd += ["--device", str(args.device)]
    return cmd


def main():
    """启动看门狗，监控训练进程并在异常退出时自动重启。"""
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--resume",
    #     type=str,
    #     default=str(default_resume_path()),
    #     help="断点路径：可传运行目录或具体 last.pt 文件；默认为上次正式训练目录",
    # )
    parser.add_argument("--imgsz", type=int, default=None, help="可选覆盖：输入尺寸")
    parser.add_argument("--batch", type=int, default=None, help="可选覆盖：批大小")
    parser.add_argument("--device", type=str, default=None, help="可选覆盖：设备，如 '0'、'0,1' 或 'cpu'")
    parser.add_argument("--max-retries", type=int, default=100, help="最大重启次数，达到上限后停止")
    parser.add_argument("--cooldown", type=int, default=5, help="重启前的冷却秒数")
    parser.add_argument("--no-launch-blocking", action="store_true", help="关闭 CUDA_LAUNCH_BLOCKING=1")
    args = parser.parse_args()

    env = os.environ.copy()
    if not args.no_launch_blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"  # 子进程内开启 CUDA 同步报错

    cmd = build_command(args)
    print("[Watchdog] 启动监控：", " ".join(cmd))
    attempt = 0
    while True:
        print(f"[Watchdog] 尝试运行训练进程（attempt={attempt}）...")
        try:
            # 使用同步运行，直接继承父进程的标准输出，便于实时观察训练日志
            result = subprocess.run(cmd, cwd=str(ROOT), env=env)
            if result.returncode == 0:
                print("[Watchdog] 训练进程正常结束，退出监控。")
                break
            else:
                print(f"[Watchdog] 训练进程异常退出（code={result.returncode}），准备重启...")
        except Exception as e:
            print(f"[Watchdog] 启动或运行训练进程时发生异常：{e!r}")

        # 重试控制
        attempt += 1
        if attempt > args.max_retries:
            print("[Watchdog] 已达到最大重试次数，停止监控。")
            break

        # 冷却期与指数退避
        wait_s = int(args.cooldown)
        print(f"[Watchdog] 冷却等待 {wait_s}s 后重试（第 {attempt} 次）。")
        time.sleep(wait_s)


if __name__ == "__main__":
    main()

