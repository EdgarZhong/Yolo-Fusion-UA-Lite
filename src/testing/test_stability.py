"""
稳定性测试脚本：验证模型在单模态缺失或干扰下的鲁棒性

功能：
- 继承通用测试脚本的评估逻辑
- 增加 `--modal-dropout` 参数，支持 rgb/ir/random 三种模式
- 在数据加载后、推理前注入模态置零逻辑
- 输出结果到 result_modal_dropout/ 目录下
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
import torch
import warnings

# ====== 保证导入本仓库随附的 ultralytics-8.2 源码 ======
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

# 复用通用测试脚本中的逻辑
# 注意：由于我们要修改 OBBValidator 的行为，可能需要继承并重写
from test_general import run_eval as run_eval_base, _find_weights, WEIGHTS_DIR, DEFAULT_RESULT_DIR, IMG_SIZE, CONF_THRES, IOU_THRES, MAX_DET, DATA_CFG, BATCH

from ultralytics.models.yolo.obb import OBBValidator
from ultralytics.utils import ops

class StabilityValidator(OBBValidator):
    """
    支持模态 Dropout 的验证器
    """
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, modal_dropout='none'):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.modal_dropout = modal_dropout
        print(f"[Stability] Initialized with modal_dropout={self.modal_dropout}")

    def preprocess(self, batch):
        """
        重写预处理：在归一化后执行模态置零
        """
        # 调用父类预处理（归一化、设备移动等）
        batch = super().preprocess(batch)
        
        # 仅处理图像张量
        if self.modal_dropout == 'none':
            return batch
            
        imgs = batch['img'] # [B, 6, H, W]
        B = imgs.shape[0]
        
        # 执行置零
        if self.modal_dropout == 'rgb':
            # 丢弃 RGB (前3通道)
            imgs[:, 0:3, :, :] = 0.0
        elif self.modal_dropout == 'ir':
            # 丢弃 IR (后3通道)
            imgs[:, 3:6, :, :] = 0.0
        elif self.modal_dropout == 'random':
            # 随机丢弃 RGB 或 IR (互斥)
            # 生成 B 个随机数
            rand = torch.rand(B, device=imgs.device)
            # < 0.5 丢弃 RGB, >= 0.5 丢弃 IR
            mask_rgb = rand < 0.5
            mask_ir = ~mask_rgb
            
            if mask_rgb.any():
                imgs[mask_rgb, 0:3, :, :] = 0.0
            if mask_ir.any():
                imgs[mask_ir, 3:6, :, :] = 0.0
        
        batch['img'] = imgs
        return batch

def run_stability_eval(
    weights: Path,
    device: str = "0",
    modal_dropout: str = "none",
    result_dir: Path | None = None,
    run_name: str | None = None,
    test_ratio: float = 1.0,
    iou: float = IOU_THRES,
) -> Path:
    
    # 结果目录
    result_root = Path(result_dir) if result_dir else DEFAULT_RESULT_DIR
    result_root.mkdir(parents=True, exist_ok=True)
    
    # 自动生成 run_name (如果未提供)
    if not run_name:
        # 默认命名：ModelName_DropoutMode
        model_name = weights.parent.parent.name if weights.parent.name == 'weights' else weights.stem
        run_name = f"{model_name}_{modal_dropout}"
        
    out_dir = result_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 组装参数
    workers = 0 if str(device).lower() == "cpu" else 2
    args = dict(
        task="obb",
        model=str(weights),
        data=str(DATA_CFG),
        split="test",
        imgsz=IMG_SIZE,
        rect=True,
        batch=BATCH,
        workers=workers,
        device=device,
        plots=False,
        save_json=True,
        conf=CONF_THRES,
        iou=float(iou),
        augment=False, # 稳定性测试不开启 TTA，专注于模态影响
        max_det=MAX_DET,
        project=str(result_root),
        name=run_name,
    )
    
    # 实例化自定义验证器
    # 注意：ultralytics 的 model.val() 或 validator() 调用比较复杂
    # 这里我们直接实例化并运行
    
    # 1. 临时修改 OBBValidator 为我们的子类？
    # 或者直接使用 validator(model=...)
    # 这里的难点是 ultralytics 内部实例化逻辑是写死的。
    # 我们可以 monkey patch 或者直接实例化 validator 并调用 __call__
    
    # 更简单的方法：直接实例化 StabilityValidator
    validator = StabilityValidator(args=args, modal_dropout=modal_dropout)
    
    # 如果有 test_ratio，处理 DataLoader (复用 test_general 逻辑)
    if isinstance(test_ratio, float) and 0.0 < test_ratio < 1.0:
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.cfg import get_cfg
        from ultralytics.data import build_yolo_dataset
        from torch.utils.data import DataLoader, Subset

        data = check_det_dataset(str(DATA_CFG))
        cfg = get_cfg(overrides=dict(task="obb", imgsz=IMG_SIZE, rect=True))
        base_ds = build_yolo_dataset(cfg, data.get("test"), BATCH, data, mode="val")
        
        count = max(1, int(len(base_ds) * test_ratio))
        indices = list(range(count))
        subset = Subset(base_ds, indices)
        loader = DataLoader(
            dataset=subset,
            batch_size=min(BATCH, len(indices)),
            shuffle=False,
            num_workers=workers,
            collate_fn=getattr(base_ds, "collate_fn", None),
            pin_memory=False,
        )
        validator.dataloader = loader
        print(f"[Stability] Created subset dataloader with {count} samples.")
        
    # 执行验证
    stats = validator(model=str(weights))
    
    # 后续保存逻辑复用 test_general.py 的代码太麻烦，这里简单重写保存
    import json
    import csv
    
    names = getattr(validator, "names", {})
    nc = getattr(validator, "nc", len(names) if isinstance(names, dict) else 0)
    per_class = []
    if nc:
        for i in range(nc):
            try:
                p_i, r_i, ap50_i, ap_i = validator.metrics.class_result(i)
            except Exception:
                p_i, r_i, ap50_i, ap_i = 0.0, 0.0, 0.0, 0.0
            cname = names.get(i, str(i)) if isinstance(names, dict) else str(i)
            per_class.append({
                "id": i, "name": cname, 
                "precision": float(p_i), "recall": float(r_i), 
                "ap50": float(ap50_i), "ap": float(ap_i)
            })

    enriched = dict(stats)
    enriched["names"] = names
    enriched["classes"] = per_class
    
    # JSON
    with open(out_dir / f"{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
        
    # CSV
    with open(out_dir / f"{run_name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in stats.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["class", "precision", "recall", "ap50", "ap"])
        for item in per_class:
            w.writerow([item["name"], item["precision"], item["recall"], item["ap50"], item["ap"]])
            
    print(f"[Stability] Results saved to {out_dir}")
    return out_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--modal-dropout", type=str, default="none", choices=['none', 'rgb', 'ir', 'random'])
    parser.add_argument("--result-dir", type=str, default="result_modal_dropout")
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--test-ratio", type=float, default=1.0)
    
    args = parser.parse_args()
    
    weights_input = Path(args.weights) if args.weights else _find_weights(WEIGHTS_DIR)
    weights = weights_input if weights_input.is_file() else _find_weights(weights_input)
    
    run_stability_eval(
        weights=weights,
        device=args.device,
        modal_dropout=args.modal_dropout,
        result_dir=Path(args.result_dir),
        run_name=args.model_name,
        test_ratio=args.test_ratio
    )

if __name__ == '__main__':
    main()
