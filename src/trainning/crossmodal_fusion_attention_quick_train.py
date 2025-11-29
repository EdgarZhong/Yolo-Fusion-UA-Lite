import sys
from pathlib import Path

# ================== 仓库根路径与 Ultralytics 源码导入 ==================
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBTrainer


def main():
    """
    CM‑FA‑Concat 快速训练脚本：用于验证模块与数据管线可正常运行

    策略：
    - 使用新模型配置 `dualbackbone_CM-FA-Concat.yaml`；
    - 使用裁切白边后的数据集 YAML；
    - 训练 1 个 epoch，`fraction=0.1` 子集加速；
    - 开启 mosaic 增强，输入分辨率统一为 640（配合数据裁切的 640x512）。
    """

    model_cfg = str(ROOT / "src" / "cfg" / "model" / "dualbackbone_CM-FA-Concat.yaml")
    data_cfg = str(ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml")

    overrides = {
        # 任务与核心路径
        "task": "obb",
        "model": model_cfg,
        "data": data_cfg,
        # 训练规模与资源（快速试跑）
        "epochs": 1,
        "batch": 8,
        "workers": 0,
        "device": 0,
        # 子集训练以加速试跑
        "fraction": 0.1,
        # 输入与增强
        "imgsz": 640,
        "rect": False,
        "mosaic": 1.0,
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
        # 输出目录与命名
        "project": str(ROOT / "models" / "fusion-attention"),
        "name": "dualbackbone-CM-FA-Concat-obb-quick",
        "save": True,
        "val": True,
        "patience": 0,
        "plots": False,
        # 明确不在尾期关闭 mosaic（默认 10），设为 0 表示始终开启
        "close_mosaic": 0,
    }

    trainer = OBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
