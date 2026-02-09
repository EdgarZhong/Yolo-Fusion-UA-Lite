
import warnings
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
from ultralytics.utils import RANK

# 导入增强型模态 Dropout Hook - 已集成到 Ultralytics 框架中，此处移除显式 Hook 注入
# from modality_dropout_hook import inject_enhanced_modality_dropout

def transfer_weights(source_path, target_model):
    """
    执行权重迁移：
    1. 加载源模型 (FA-Concat-Tuned)
    2. 筛选并加载 Backbone, Neck, Head 和 Fusion.Inception
    3. 丢弃不兼容的 SE 权重，保留 CM-SE 为随机初始化
    """
    if RANK in (-1, 0):
        print(f"\n[Transfer] Loading source weights from: {source_path}")
    
    # 加载源权重字典
    if isinstance(source_path, (str, Path)):
        ckpt = torch.load(str(source_path), map_location='cpu')
        source_sd = ckpt['model'].float().state_dict()
    else:
        source_sd = source_path.float().state_dict()
        
    target_sd = target_model.state_dict()
    
    transferred_keys = []
    skipped_keys = []
    
    for k, v in source_sd.items():
        # 必须是目标模型中存在的键
        if k not in target_sd:
            continue
            
        # 排除形状不匹配的层 (主要是不兼容的 SE 模块)
        # FA-Concat SE: c -> c/16 -> c
        # CM-FA SE: 2c -> c/8 -> 2c
        if v.shape != target_sd[k].shape:
            skipped_keys.append(k)
            continue
            
        # 排除包含 'se_rgb' 或 'se_ir' 的键 (显式丢弃旧 SE)
        # 虽然形状检查可能已经排除了，但显式过滤更安全
        if 'se_rgb' in k or 'se_ir' in k:
            skipped_keys.append(k)
            continue
            
        # 执行加载
        with torch.no_grad():
            target_sd[k].copy_(v)
        transferred_keys.append(k)
        
    if RANK in (-1, 0):
        print(f"[Transfer] Successfully transferred {len(transferred_keys)} layers.")
        print(f"[Transfer] Skipped/Initialized {len(skipped_keys)} layers (Incompatible shapes or explicit skip).")
        # print(f"[Transfer] Example skipped: {skipped_keys[:5]}")

def freeze_layers(model, freeze_mode='init'):
    """
    冻结策略：
    - init: 冻结 Backbone (0-22) 和 Inception。解冻 CM-SE 和 Neck/Head (23+)。
      (修正：原策略冻结 Neck/Head 导致特征不匹配无法收敛，必须解冻下游层以适配新特征)
    - unfreeze: 解冻所有层
    """
    if freeze_mode == 'init':
        print("[Freeze] Locking Backbone (0-22) & Inception. Training CM-SE and Neck/Head.")
        
        # 假设 model 是 DetectionModel，model.model 是 nn.Sequential
        # 遍历所有层，根据索引判断是否为 Backbone
        for i, m in enumerate(model.model):
            # 0-22 为 Backbone (根据 YAML 配置)
            if i <= 22:
                # 冻结 Backbone
                for param in m.parameters():
                    param.requires_grad = False
            else:
                # 23+ 为 Neck/Head (包含 Fusion 层, FPN, PAN, Head)
                # 默认解冻，允许下游层适配新的 Fusion 输出
                for param in m.parameters():
                    param.requires_grad = True
                
                # 对于 Fusion 层 (CrossModalFusionAttention)，特殊处理 Inception
                # Inception 权重已迁移，建议冻结以保持特征提取稳定性
                if hasattr(m, 'inc_rgb'): 
                     for param in m.inc_rgb.parameters():
                         param.requires_grad = False
                if hasattr(m, 'inc_ir'):
                     for param in m.inc_ir.parameters():
                         param.requires_grad = False
                
                # 确保 cm_se 是解冻的 (它是新初始化的，必须训练)
                if hasattr(m, 'cm_se'):
                    for param in m.cm_se.parameters():
                        param.requires_grad = True
                        
    elif freeze_mode == 'unfreeze':
        print("[Freeze] Unfreezing all layers for fine-tuning.")
        # 注意：model.named_parameters() 返回的是 (name, param) 元组
        for name, param in model.named_parameters():
            param.requires_grad = True

def main():
    # -------------------------------------------------------------------------
    # 1. 配置路径
    # -------------------------------------------------------------------------
    ROOT = Path(__file__).resolve().parents[2] # YOLO-Fusion-UA-Lite/
    
    # 源权重 (M5: FA-Concat-Tuned)
    SOURCE_WEIGHTS = ROOT / "models" / "posttrain" / "FA-Concat_FPN-PAN_tuned" / "weights" / "best.pt"
    
    # 目标配置
    TARGET_CFG = ROOT / "src" / "cfg" / "model" / "CM-FA_Concat_FPN-PAN_neck.yaml"
    DATA_YAML = ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml"
    
    # 输出目录
    PROJECT_DIR = ROOT / "models" / "posttrain"
    NAME = "CM-FA-Transferred-3"
    
    if not SOURCE_WEIGHTS.exists():
        raise FileNotFoundError(f"Source weights not found at: {SOURCE_WEIGHTS}")

    # -------------------------------------------------------------------------
    # 2. 构建目标模型
    # -------------------------------------------------------------------------
    # 使用 YAML 构建未初始化的模型
    model = YOLO(TARGET_CFG)
    
    # -------------------------------------------------------------------------
    # 3. 权重迁移
    # -------------------------------------------------------------------------
    # 注意：YOLO() 构造时可能已经随机初始化了
    # 我们需要在 train() 之前手动注入权重，或者利用 load()
    # 但由于结构不同，load() 会报错，所以我们手动迁移
    transfer_weights(SOURCE_WEIGHTS, model.model)
    
    # -------------------------------------------------------------------------
    # 4. 定义训练回调 (Callbacks) 实现冻结策略
    # -------------------------------------------------------------------------
    def on_train_start(trainer):
        # 初始阶段：仅训练 CM-SE
        freeze_layers(trainer.model, freeze_mode='init')
        
    def on_train_epoch_start(trainer):
        # 第 10 Epoch 解冻全网
        if trainer.epoch == 10:
            freeze_layers(trainer.model, freeze_mode='unfreeze')

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)

    # -------------------------------------------------------------------------
    # 5. 注入增强型模态 Dropout
    # -------------------------------------------------------------------------
    # 模态 Dropout 现已集成到 ultralytics/models/yolo/detect/train.py 中
    # 通过 model.train() 参数直接控制，不再需要 Hook 注入

    # -------------------------------------------------------------------------
    # 6. 启动训练
    # -------------------------------------------------------------------------
    
    # 基础设置
    USE_TEST_AS_VAL = True
    ADD_VAL_TO_TRAIN = True

    # 参数复用 FA-Concat-Tuned 的成功经验
    model.train(
        data=DATA_YAML,
        project=PROJECT_DIR,
        name=NAME,
        epochs=50,          # 修正：从 30 增加回 50
                            # 原因：前 10 Epoch 处于"移植排异期"（Adaptation Phase），模型在努力适应新模块，
                            # 性能处于低位。若仅训练 30 轮，留给全解冻后的微调时间仅 20 轮，可能导致欠拟合。
                            # 50 轮能保证有充足的 fine-tuning 窗口。
        batch=16,            # 显存允许下的 Batch
        imgsz=640,          # 裁切数据标准尺寸
        rect=False,         # 训练时不使用矩形 (因为是 640x512 裁切图，直接 resize 到 640 即可，或者 rect=False 配合多尺度)
                            # 修正：之前经验是训练 rect=False, 验证 rect=True
        
        # 优化器与学习率
        optimizer='SGD',
        # 修正：LR0 从 0.002 提升至 0.005。
        # 原因：前 10 Epoch 仅训练小参数量的 CM-SE 模块，0.002 导致收敛过慢（mAP~0.0006）。
        # 0.005 能加速初期适配，配合 lrf=0.01 最终衰减至 5e-5，仍保证微调安全。
        lr0=0.005,          
        lrf=0.01,           
        warmup_epochs=3.0,
        
        # 数据增强 (复用 FA-Concat-Tuned)
        mosaic=1.0,         # 开启 Mosaic
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,        # OBB 敏感
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.0,          # 保持红外特征
        hsv_s=0.0,
        hsv_v=0.0,
        
        # 其他
        workers=4,
        close_mosaic=8,     # 修正：最后 8 epoch 关闭 Mosaic
        device='0',
        exist_ok=True,
        save=True,
        val=True,           # 训练中验证
        max_det=200,        # 减小最大检测框数量
        plots=True,
        
        # 强制使用测试集进行验证 (重要)
        # 根据项目约定，use_test_as_val=True 会在训练期间使用测试集作为验证集
        # 这通常需要修改过的 Ultralytics 源码支持，或者作为自定义参数传递给 Trainer
        use_test_as_val=USE_TEST_AS_VAL,
        
        # [新增] 将验证集加入训练集
        add_val_to_train=ADD_VAL_TO_TRAIN,
        
        # 模态随机失活 (Modality Dropout)
        drop_prob_rgb=0.2,  # RGB 模态丢失概率
        drop_prob_ir=0.2,   # IR 模态丢失概率
        close_dropout=10    # 最后 10 epoch 关闭 Dropout
    )

if __name__ == '__main__':
    main()
