import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from ultralytics.models.yolo.obb import OBBTrainer


def main():
    model_cfg = str(ROOT / "src" / "cfg" / "model" / "dualbackbone_fusionattention_obb.yaml")
    data_cfg = str(ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml")

    overrides = {
        "task": "obb",
        "model": model_cfg,
        "data": data_cfg,
        "epochs": 1,
        "batch": 8,
        "workers": 2,
        "device": 0,
        "fraction": 0.05,
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
        "project": str(ROOT / "models" / "fusion-attention"),
        "name": "dualbackbone-fusionattention-obb-baseline",
        "save": True,
        "val": True,
        "patience": 0,
        "plots": False,
    }

    trainer = OBBTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
