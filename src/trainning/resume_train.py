"""
断点续训脚本（OBB 单/双模态自适配）

设计目标：
- 支持从上次正式训练运行目录或手动指定路径恢复。
- 自动识别检查点的输入通道数（3=IR 单模态；6=双主干融合），并构建匹配的模型架构，避免权重/EMA维度错配。
- 保持验证器关键参数与双模态训练一致以便对比（如 iou/conf/max_det 等保持默认一致）。
- 仅允许覆盖 `imgsz/batch/device/freeze` 四项以便迁移与资源调整，其余参数从检查点恢复。
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
from ultralytics.nn.tasks import OBBModel  # noqa: E402
from ultralytics.nn.tasks import torch_safe_load  # noqa: E402


def default_resume_path() -> Path:
    """返回默认 last.pt 路径：指向 IR‑YOLOv8n 单模态训练输出的 last.pt。"""
    return ROOT / "models" / "IR-YOLOv8n" / "from_scrach" / "train" / "weights" / "last.pt"


class IR_OBB_Trainer(OBBTrainer):
    """IR 单模态 OBB 训练器，仅重写 get_model 构建 3 通道原版 OBB 模型，并禁止预训练权重加载。"""

    def get_model(self, cfg=None, weights=None, verbose=True):
        _ = None
        model = OBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose)
        # 恢复训练场景：若传入权重（.pt），需加载到模型，确保 EMA/优化器后续正确恢复
        if weights:
            model.load(weights)
        return model


def resolve_last_pt(resume_arg: str | Path) -> Path:
    """将传入的 --resume 参数解析为具体 last.pt 文件路径。支持传入目录或 .pt 文件。"""
    p = Path(resume_arg)
    if p.is_dir():
        # 优先寻找目录下的 weights/last.pt
        cand = p / "weights" / "last.pt"
        if cand.exists():
            return cand
        # 其次直接寻找目录内的 .pt 文件
        pts = sorted(p.rglob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if pts:
            return pts[0]
        raise FileNotFoundError(f"目录下未找到 .pt 文件：{p}")
    if p.suffix == ".pt" and p.exists():
        return p
    raise FileNotFoundError(f"无效的 --resume 路径：{p}")


def infer_in_channels(ckpt_pt: Path) -> int:
    """读取检查点的模型首层卷积权重维度，推断输入通道数（3 或 6）。"""
    ckpt, _ = torch_safe_load(str(ckpt_pt))
    m = ckpt.get("ema") or ckpt["model"]
    try:
        return int(m.model[0].conv.weight.shape[1])
    except Exception:
        # 回退：从 state_dict 键检索
        sd = (ckpt.get("ema") or ckpt["model"]).state_dict()
        for k in ("model.0.conv.weight", "model.0.m.0.weight"):
            w = sd.get(k)
            if w is not None:
                return int(w.shape[1])
        raise RuntimeError("无法从检查点推断输入通道数")


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
    parser.add_argument("--freeze", type=int, default=0, help="可选覆盖：按层冻结前 N 层（设 0 关闭冻结）")
    args = parser.parse_args()

    # 解析实际 last.pt 文件并推断输入通道数
    ckpt_pt = resolve_last_pt(args.resume)
    in_ch = infer_in_channels(ckpt_pt)

    overrides = {
        "task": "obb",
        "resume": str(ckpt_pt),
        # 验证器对齐关键参数（与双模态一致的默认值）
        "iou": 0.7,
        "max_det": 300,
        "plots": False,
        "val": True,
    }

    # 仅允许覆盖 imgsz/batch/device 三项（其余由检查点决定）
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.device is not None:
        overrides["device"] = args.device
    if args.freeze is not None:
        overrides["freeze"] = args.freeze

    # 运行信息提示
    print("[Resume][OBB] 断点续训启动：")
    print(f"resume={ckpt_pt}")
    print(f"detected_in_channels={in_ch}")

    # 强制开启验证和绘图，确保 Loss 被计算和记录
    overrides["save"] = True

    if args.imgsz is not None:
        print(f"override imgsz={args.imgsz}")
    if args.batch is not None:
        print(f"override batch={args.batch}")
    if args.device is not None:
        print(f"override device={args.device}")
    if args.freeze is not None:
        print(f"override freeze={args.freeze}")

    # 根据检查点架构自适配训练器：3通道→IR 单模态；6通道→双主干默认实现
    trainer = IR_OBB_Trainer(overrides=overrides) if in_ch == 3 else OBBTrainer(overrides=overrides)
    # 由于 Ultralytics 的 check_resume 仅允许覆盖 imgsz/batch/device，这里在实例化后强制覆盖 freeze
    if args.freeze is not None:
        trainer.args.freeze = args.freeze
    trainer.train()


if __name__ == "__main__":
    main()
