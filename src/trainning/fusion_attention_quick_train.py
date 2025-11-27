import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBTrainer


def main():
    # 使用新的 FA‑Concat 模型配置进行快速训练验证
    model_cfg = str(ROOT / "src" / "cfg" / "model" / "dualbackbone_FA-Concat.yaml")
    data_cfg = str(ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml")

    overrides = {
        "task": "obb",
        "model": model_cfg,
        "data": data_cfg,
        "epochs": 1,
        "batch": 10,
        "workers": 2,
        "device": 0,
        "fraction": 0.1,
        "imgsz": 832,
        "rect": False,
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
        # 输出目录与命名：保持原目录，名称更新以便区分本次实验
        "project": str(ROOT / "models" / "fusion-attention"),
        "name": "dualbackbone-FA-Concat-obb-quick",
        "save": True,
        "val": True,
        "patience": 0,
        "plots": False,
    }

    trainer = OBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
