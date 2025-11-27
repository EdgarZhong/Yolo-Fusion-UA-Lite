"""
断点续训脚本（OBB 双主干 + FusionAttention）

设计目标：
- 默认从上次正式训练运行目录恢复：`models/formal/fusion-attention/dualbackbone-fusionattention-obb/weights/last.pt`
- 允许通过命令行传入自定义 `--resume` 路径（指向 last.pt 文件或运行目录），若未找到则给出明确提示。
- 保持 AMP 与 batch 等配置由检查点决定，仅允许覆盖 `imgsz/batch/device` 三项以便迁移与资源调整。

"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 开启 CUDA 调试模式，及时捕获异常
import argparse
import sys
from pathlib import Path

# 保证使用仓库内 ultralytics-8.2 源码
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBTrainer  # noqa: E402


def default_resume_path() -> Path:
    """返回默认的 last.pt 路径，如果不存在则返回运行目录路径用于框架内部查找。"""
    run_dir = ROOT / "models" / "formal" / "dualbackbone-FA-Concat-obb"
    last_pt = run_dir / "weights" / "last.pt"
    return last_pt if last_pt.exists() else run_dir


def main():
    """启动断点续训：只设置 resume，其他参数交由检查点恢复，必要时允许覆盖 imgsz/batch/device。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=str(default_resume_path()),
        help="断点路径：可传运行目录或具体 last.pt 文件；默认为上次正式训练目录",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="可选覆盖：输入尺寸")
    parser.add_argument("--batch", type=int, default=None, help="可选覆盖：批大小")
    parser.add_argument("--device", type=str, default=None, help="可选覆盖：设备，如 '0'、'0,1' 或 'cpu'")
    args = parser.parse_args()

    overrides = {
        "task": "obb",
        "resume": args.resume,
    }

    # 仅允许覆盖 imgsz/batch/device 三项（其余由检查点决定）
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.device is not None:
        overrides["device"] = args.device

    # 运行信息提示
    print("[Resume][OBB] 断点续训启动：")
    print(f"resume={args.resume}")

    # 强制开启验证和绘图，确保 Loss 被计算和记录
    overrides["val"] = True
    overrides["save"] = True

    if args.imgsz is not None:
        print(f"override imgsz={args.imgsz}")
    if args.batch is not None:
        print(f"override batch={args.batch}")
    if args.device is not None:
        print(f"override device={args.device}")

    trainer = OBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()

