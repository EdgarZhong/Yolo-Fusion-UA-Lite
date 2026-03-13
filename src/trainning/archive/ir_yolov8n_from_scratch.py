"""
原版 YOLOv8n OBB 模型在 IR 图像上的从零训练脚本

目标与约束：
- 使用原始（含白边）的数据集，目录位于仓库 `data/` 下；只加载 IR 单模态图像（`*imgr/`）。
- 不加载预训练权重：从随机初始化开始构建网络参数。
- 使用默认超参，但关闭所有数据增强（mosaic/mixup/拷贝粘贴/擦除/颜色/仿射/翻转等）。
- 训练 100 轮；脚本提供参数可覆盖（便于快速 sanity 验证）。
- 与现有双模态训练完全解耦：不修改框架训练器，只在脚本内自定义 Trainer。
- 训练输出存放到 `models/IR-YOLOv8n/from_scrach/` 目录下（按用户要求的路径命名）。

设计要点：
- 数据加载解耦：通过数据集 YAML 指向 `*imgr/` 目录，构建器自动选择 `YOLOIRDataset`，不会触发双模态逻辑。
- 模型构建解耦：自定义 `IR_OBB_Trainer` 继承官方 `OBBTrainer`，仅重写 `get_model`，返回 3 通道的 `OBBModel`，不加载权重。
- 增强关闭：通过 `overrides` 将所有增强超参置零；训练阶段仍走标准管线，但增强算子全部失效。
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

# 保障本地 ultralytics 源码可被导入（避免外部环境未安装）
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

# 导入 OBB 训练器与模型构建工具
from ultralytics.models.yolo.obb import OBBTrainer  # noqa: E402
from ultralytics.nn.tasks import OBBModel  # noqa: E402


class IR_OBB_Trainer(OBBTrainer):
    """
    IR 单模态 OBB 训练器：
    - 仅重写 `get_model`，构建 3 通道输入的原版 YOLOv8n OBB 模型；不加载预训练权重。
    - 其余训练/验证逻辑复用官方/现有实现，确保与双模态训练完全解耦。
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        构建 3 通道原版 YOLOv8n OBB 模型：
        - `cfg`：模型 YAML（包含 `scales`，自动根据文件名推断 `n/s/m/...`，未指定则默认 `n`）。
        - `weights`：此处必须为 None（从零训练），若传入将忽略不加载。
        - `verbose`：是否打印模型结构信息。
        """
        # 明确禁止预训练权重加载：从零开始
        _ = None  # 占位以强调不使用权重
        model = OBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose)
        return model


def build_overrides(
    data_yaml: Path,
    model_yaml: Path,
    project_dir: Path,
    device: str = "0",
    epochs: int = 100,
    batch: int = 12,
    imgsz: int = 640,
):
    """
    构建训练参数覆盖字典（overrides），关闭全部数据增强。

    参数：
    - `data_yaml`：数据集 YAML（指向 data/ 下的 `*imgr/` 目录）。
    - `model_yaml`：模型 YAML（使用 `yolov8n-obb.yaml` 名称以选择 `n` 尺度）。
    - `project_dir`：训练输出目录（`models/IR-YOLOv8n/from_scrach`）。
    - 其余为训练资源与输入尺寸参数。
    """
    # GPU/CPU 线程：CPU 设置为 0，GPU 默认 2
    workers = 0 if str(device).lower() in {"cpu", "mps"} else 2

    return {
        # 任务与核心路径
        "task": "obb",
        "model": str(model_yaml),
        "data": str(data_yaml),
        # 训练规模与资源
        "epochs": int(epochs),
        "batch": int(batch),
        "workers": int(workers),
        "device": device,
        # 输入与评估配置（保持默认 img 大小，训练阶段非矩形；验证阶段内部自动矩形）
        "imgsz": int(imgsz),
        "rect": False,
        # 关闭所有数据增强
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 1.0,
        "shear": 0.0,
        "perspective": 0.0,
        # 保存位置（严格按用户要求路径）
        "project": str(project_dir),
        # 训练过程选项
        "save": True,
        "val": True,
        "patience": 0,
        "plots": False,
    }


def main():
    """
    命令行入口：
    - 默认运行 100 轮无增强训练；如需快速验证，可通过参数覆盖。
    - 示例：
      - GPU 快速 sanity：`python src/trainning/ir_yolov8n_from_scratch.py --device 0 --epochs 1 --batch 8 --imgsz 640`
      - CPU 小集验证：`python src/trainning/ir_yolov8n_from_scratch.py --device cpu --epochs 1 --batch 4`
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=832)
    args_ns = parser.parse_args()

    # 路径：数据集 YAML（原始 IR）、模型 YAML（n 尺度）、输出目录（from_scrach）
    data_yaml = ROOT / "src/cfg/datasets/ir_obb_dronevehicle_raw.yaml"
    model_yaml = ULTRA / "ultralytics/cfg/models/v8/yolov8n-obb.yaml"
    project_dir = ROOT / "models/IR-YOLOv8n/from_scrach"

    overrides = build_overrides(
        data_yaml=data_yaml,
        model_yaml=model_yaml,
        project_dir=project_dir,
        device=args_ns.device,
        epochs=args_ns.epochs,
        batch=args_ns.batch,
        imgsz=args_ns.imgsz,
    )

    print(
        f"[Train][IR-OBB] 从零训练：epochs={overrides['epochs']}, batch={overrides['batch']}, imgsz={overrides['imgsz']}, device={overrides['device']}\n"
        f"[Train][IR-OBB] 数据配置: {data_yaml}\n"
        f"[Train][IR-OBB] 模型配置: {model_yaml}\n"
        f"[Train][IR-OBB] 输出目录: {project_dir}"
    )

    trainer = IR_OBB_Trainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()

