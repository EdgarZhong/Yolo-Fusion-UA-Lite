"""
IR 单模态 YOLOv8n OBB 检测模型训练脚本（COCO 预训练权重初始化）

本脚本作为双模态融合模型的单模态性能基准，使用：
- 标准 YOLOv8n 架构（无自定义融合模块）
- 单模态 IR 输入（3 通道）
- COCO 官方预训练权重 yolov8n-obb.pt 初始化

技术实现：通过自定义 IR_OBB_Trainer 重写 get_model() 方法，
使用标准 OBBModel(ch=3) 替代 DualBackboneOBBModel(ch=6)。
"""

import sys
from pathlib import Path

# ================== 仓库根路径与 Ultralytics 源码导入 ==================
ROOT = Path(__file__).resolve().parents[2]
ULTRA = ROOT / "ultralytics-8.2"
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

# 导入 OBB 训练器与模型构建工具
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import OBBModel

# 训练运行名称，用于确定输出目录
RUN_NAME = "IR-Only-Pretrained"
# 总训练轮次
TOTAL_EPOCHS = 160
# 训练尾期关闭 Mosaic 的轮次数
CLOSE_MOSAIC_EPOCHS = 16
# 冻结 backbone 的轮次数
FREEZE_EPOCHS = 10


class IR_OBB_Trainer(OBBTrainer):
    """
    IR 单模态 OBB 训练器：
    - 重写 get_model，构建 3 通道输入的标准 YOLOv8n OBB 模型
    - 加载 COCO 预训练权重 yolov8n-obb.pt 进行初始化
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        构建 3 通道标准 YOLOv8n OBB 模型，并加载预训练权重
        """
        # 使用标准 OBBModel，3 通道输入
        model = OBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose)
        # 加载预训练权重（如果提供）
        if weights:
            model.load(weights)
        return model


def _root_dir() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parents[2]


def get_run_dir() -> Path:
    """获取训练输出目录（用于 Train Manager）"""
    root = _root_dir()
    # 快速验证时输出到 models/baseline，全量训练后用户可手动移动到 models/posttrain
    project_dir = root / "models" / "baseline"
    return project_dir / RUN_NAME


def build_train_command() -> list[str]:
    """构建训练命令（用于 Train Manager）"""
    return [sys.executable, str(Path(__file__).resolve())]


def build_resume_command() -> list[str]:
    """构建恢复训练命令（用于 Train Manager）"""
    root = _root_dir()
    run_dir = get_run_dir()
    return [
        sys.executable,
        str(root / "src" / "trainning" / "resume_train.py"),
        "--resume",
        str(run_dir),
    ]


def get_train_manager_spec() -> dict:
    """
    返回 Train Manager 规范配置

    Returns:
        dict: 包含训练命令、恢复命令、输出目录等信息的字典
    """
    run_dir = get_run_dir()
    return {
        "name": "ir_only_pretrained_baseline",
        "train_cmd": build_train_command(),
        "resume_cmd": build_resume_command(),
        "resume_ready": str(run_dir / "weights" / "last.pt"),
        "workdir": str(_root_dir()),
        "run_dir": str(run_dir),
        "total_epochs": TOTAL_EPOCHS,
    }


def main():
    """
    主训练函数

    使用 YOLOv8n COCO 预训练权重初始化，在 IR 单模态数据上训练 OBB 检测模型。
    这是双模态融合模型的性能基准实验。
    """
    # -------------------------------------------------------------------------
    # 配置路径与参数
    # -------------------------------------------------------------------------
    # COCO 预训练权重（YOLOv8n-obb.pt），ultralytics 会自动下载如果不存在
    # 注意：必须使用 -obb 版本，OBB 任务使用旋转框格式（xywhr），与标准检测（xyxy）不同
    PRETRAINED_WEIGHTS = "yolov8n-obb.pt"

    # 数据集配置（IR 单模态，指向 data_croped/ 目录）
    DATA_YAML = ROOT / "src" / "cfg" / "datasets" / "ir_obb_dronevehicle.yaml"

    # 输出目录配置
    RUN_DIR = get_run_dir()
    PROJECT_DIR = RUN_DIR.parent
    NAME = RUN_DIR.name

    # -------------------------------------------------------------------------
    # 构建训练参数覆盖字典
    # -------------------------------------------------------------------------
    overrides = {
        # 任务与核心路径
        "task": "obb",
        "model": PRETRAINED_WEIGHTS,        # 预训练权重作为模型配置
        "data": str(DATA_YAML),
        # 训练规模与资源
        "epochs": TOTAL_EPOCHS,             # 总训练轮次
        "batch": 16,                        # 批大小
        "imgsz": 640,                       # 输入图像尺寸
        "device": 0,                        # GPU 设备索引
        "workers": 2,                       # 数据加载线程数
        # 训练策略
        "patience": 0,                      # 关闭早停（0 表示不启用）
        "freeze": FREEZE_EPOCHS,            # 冻结 backbone 前 10 轮
        "rect": False,                      # 使用方形输入（非矩形批次）
        # 验证参数
        "conf": 0.25,                       # 验证时置信度阈值
        "iou": 0.75,                        # NMS IoU 阈值
        "max_det": 300,                     # 每图最大检测数
        # 数据增强
        "mosaic": 1.0,                      # 启用 Mosaic 增强
        "close_mosaic": CLOSE_MOSAIC_EPOCHS,  # 最后 16 轮关闭 Mosaic
        "mixup": 0.0,                       # 关闭 MixUp
        "copy_paste": 0.0,                  # 关闭 Copy-Paste
        "degrees": 0.0,                     # 关闭旋转增强
        "translate": 0.1,                   # 平移增强
        "scale": 0.5,                       # 缩放增强
        "shear": 0.0,                       # 关闭剪切增强
        "perspective": 0.0,                 # 关闭透视增强
        "fliplr": 0.5,                      # 水平翻转
        "flipud": 0.0,                      # 关闭垂直翻转
        "hsv_h": 0.0,                       # 关闭 HSV 色调增强
        "hsv_s": 0.0,                       # 关闭 HSV 饱和度增强
        "hsv_v": 0.0,                       # 关闭 HSV 明度增强
        # 优化器配置（YOLOv8 默认 SGD）
        "optimizer": "SGD",
        "lr0": 0.01,                        # 初始学习率
        "lrf": 0.01,                        # 最终学习率比例
        "momentum": 0.937,                  # 动量
        "weight_decay": 0.0005,             # 权重衰减
        "warmup_epochs": 3.0,               # Warmup 轮次
        # 验证配置（使用测试集作为验证集）
        "use_test_as_val": True,
        # 输出与保存配置
        "project": str(PROJECT_DIR),        # 训练输出根目录
        "name": NAME,                       # 训练运行名称
        "exist_ok": True,                   # 允许同名目录继续训练
        "save": True,                       # 保存权重文件
        "val": True,                        # 训练期间启用验证
        "plots": True,                      # 启用图表生成
        # 其他配置
        "deterministic": True,              # 确定性训练（可复现）
        "seed": 0,                          # 随机种子
    }

    print(f"[Train][IR-OBB] 启动训练：epochs={overrides['epochs']}, batch={overrides['batch']}, imgsz={overrides['imgsz']}")
    print(f"[Train][IR-OBB] 数据配置: {DATA_YAML}")
    print(f"[Train][IR-OBB] 预训练权重: {PRETRAINED_WEIGHTS}")
    print(f"[Train][IR-OBB] 输出目录: {RUN_DIR}")

    # -------------------------------------------------------------------------
    # 启动训练
    # -------------------------------------------------------------------------
    trainer = IR_OBB_Trainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
