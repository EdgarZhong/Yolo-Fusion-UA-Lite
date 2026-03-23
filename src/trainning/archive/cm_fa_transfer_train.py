
import warnings
import sys
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
from ultralytics.utils import RANK

# 训练运行名，训练输出目录与 Train Manager 统一由此派生，避免手动维护两处
RUN_NAME = "CM-FA-Transferred-220"
# 总训练轮次，训练与 Train Manager 均由此读取，避免轮次不一致
TOTAL_EPOCHS = 220
# 训练尾期关闭 Mosaic/模态失活的轮次数，统一复用避免重复维护
FINAL_CLOSE_EPOCHS = 30
# 是否使用测试集作为验证集，统一为单一开关便于复现实验设置
USE_TEST_AS_VAL = True

# 导入增强型模态 Dropout Hook - 已集成到 Ultralytics 框架中，此处移除显式 Hook 注入
# from modality_dropout_hook import inject_enhanced_modality_dropout

def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]

def get_run_dir() -> Path:
    root = _root_dir()
    project_dir = root / "models" / "posttrain"
    return project_dir / RUN_NAME

def build_train_command() -> list[str]:
    return [sys.executable, str(Path(__file__).resolve())]

def build_resume_command() -> list[str]:
    root = _root_dir()
    run_dir = get_run_dir()
    return [
        sys.executable,
        str(root / "src" / "trainning" / "resume_train.py"),
        "--resume",
        str(run_dir),
    ]

def get_train_manager_spec() -> dict:
    run_dir = get_run_dir()
    return {
        "name": "cm_fa_transfer",
        "train_cmd": build_train_command(),
        "resume_cmd": build_resume_command(),
        "resume_ready": str(run_dir / "weights" / "last.pt"),
        "workdir": str(_root_dir()),
        "run_dir": str(run_dir),
        "total_epochs": TOTAL_EPOCHS,
    }

def transfer_weights(source_path, target_model):
    """
    执行权重迁移：
    1. 加载源模型 YOLOV8n
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

def main():
    # -------------------------------------------------------------------------
    # 1. 配置路径
    # -------------------------------------------------------------------------
    ROOT = Path(__file__).resolve().parents[2] # YOLO-Fusion-UA-Lite/
    
    # 源权重（官方 COCO 预训练权重，用于主干初始化）
    # 说明：这里改为加载官方权重，避免再从 FA-Concat tuned 权重迁移
    SOURCE_WEIGHTS = ROOT / "yolov8n.pt"
    
    # 目标配置
    TARGET_CFG = ROOT / "src" / "cfg" / "model" / "CM-FA_Concat_FPN-PAN_neck.yaml"
    DATA_YAML = ROOT / "src" / "cfg" / "datasets" / "dual_obb_dronevehicle.yaml"
    
    # 输出目录
    RUN_DIR = get_run_dir()
    PROJECT_DIR = RUN_DIR.parent
    NAME = RUN_DIR.name
    
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
    # 4. 启动训练（完全复刻历史最佳模型经验，迁移不启用冻结）
    # -------------------------------------------------------------------------
    
    # 参数完全复刻 FA-Concat_FPN-PAN_tuned 的训练经验
    model.train(
        data=DATA_YAML,                      # 数据集配置
        project=PROJECT_DIR,                 # 训练输出根目录
        name=NAME,                           # 训练运行名称
        epochs=TOTAL_EPOCHS,                 # 总训练轮次，统一与 Train Manager 保持一致
        patience=0,                          # 关闭早停机制
        freeze=10,                           # 历史最佳策略：冻结前 10 轮 Backbone
        batch=16,                            # 历史最佳模型批大小
        imgsz=640,                           # 历史最佳模型输入尺寸
        rect=False,                          # 历史最佳模型为非矩形训练
        optimizer='SGD',                     # 历史最佳模型优化器
        lr0=0.01,                            # 历史最佳模型初始学习率
        lrf=0.01,                            # 历史最佳模型末端学习率比例
        warmup_epochs=3.0,                   # 历史最佳模型 warmup 轮次
        momentum=0.937,                      # 历史最佳模型动量
        weight_decay=0.0005,                 # 历史最佳模型权重衰减
        workers=2,                           # 历史最佳模型数据加载线程
        deterministic=True,                  # 历史最佳模型确定性训练
        seed=0,                              # 历史最佳模型随机种子
        mosaic=1.0,                          # 按你的要求启用 Mosaic，并在最后 16 轮关闭
        mixup=0.0,                           # 历史最佳模型关闭 MixUp
        copy_paste=0.0,                      # 历史最佳模型关闭 Copy-Paste
        degrees=0.0,                         # 历史最佳模型旋转增强关闭
        translate=0.1,                       # 历史最佳模型平移增强
        scale=0.5,                           # 历史最佳模型缩放增强
        shear=0.0,                           # 历史最佳模型剪切增强关闭
        perspective=0.0,                     # 历史最佳模型透视增强关闭
        fliplr=0.5,                          # 历史最佳模型水平翻转
        flipud=0.0,                          # 历史最佳模型垂直翻转关闭
        hsv_h=0.0,                           # 历史最佳模型 HSV-H 关闭
        hsv_s=0.0,                           # 历史最佳模型 HSV-S 关闭
        hsv_v=0.0,                           # 历史最佳模型 HSV-V 关闭
        close_mosaic=FINAL_CLOSE_EPOCHS,     # 末尾若干轮关闭 Mosaic，统一由配置常量控制
        device='0',                          # 训练设备
        exist_ok=True,                       # 允许同名目录继续训练
        save=True,                           # 保存权重
        val=True,                            # 训练期验证开启
        iou=0.75,                            # 按你的要求设置验证 NMS IoU=0.75
        conf=0.25,                           # 按你的要求设置验证置信度阈值=0.25
        max_det=1000,                        # 增加最大检测数，以支持更多复杂高密度场景
        plots=True,                          # 启用绘图，以便后续分析
        # 迁移训练扩展参数（按你的要求启用）
        use_test_as_val=USE_TEST_AS_VAL,     # 训练期使用测试集验证（统一开关）
        drop_prob_rgb=0.2,                   # 模态随机失活：RGB 概率
        drop_prob_ir=0.2,                    # 模态随机失活：IR 概率
        close_dropout=FINAL_CLOSE_EPOCHS     # 末尾若干轮关闭模态失活，统一由配置常量控制
    )

if __name__ == '__main__':
    main()
