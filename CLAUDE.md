# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供在本仓库中工作的指导信息。

## 项目概述

**YOLO-Fusion-UA-Lite** 是一个基于修改版 Ultralytics YOLOv8-OBB 框架的双模态（RGB+IR）有向目标检测项目。核心创新在于融合模块架构，用于结合处理 RGB 和红外图像的两个并行主干网络提取的特征。

**关键技术特征：**
- **任务类型**：YOLO-OBB（旋转框检测），5 个车辆类别
- **输入格式**：6 通道张量（RGB=通道 0-2，IR=通道 3-5）
- **框架**：修改版 `ultralytics-8.2`（可编辑模式安装，非 PyPI 版本）
- **数据集**：DroneVehicle（裁剪至 640×512，去除白边）
- **硬件环境**：RTX 4060 8GB，CUDA 12.4

## 环境配置

```bash
# PowerShell - 初始化 conda
& "C:\DevLib\miniconda3\Scripts\conda.exe" shell.powershell hook | Out-String | Invoke-Expression

# 激活项目环境
conda activate .\.conda\ultra82-py312

# 验证 GPU
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

**固定版本：**
- `ultralytics==8.2.103`（来自本地 `ultralytics-8.2/`）
- `torch==2.6.0+cu124`
- `numpy==1.26.4`

## 常用开发命令

### 训练

训练通过定义超参数和模型配置的独立训练脚本执行：

```bash
# 直接运行训练脚本
python src/trainning/cm_fa_transfer_train.py

# 使用 Train Manager 恢复训练
python src/trainning/train_manager.py --script src/trainning/cm_fa_transfer_train.py --resume

# 快速冒烟测试（使用 5% 数据）
python src/trainning/crossmodal_fusion_attention_quick_train.py
```

**训练脚本约定：** 每个训练脚本导出 `get_train_manager_spec()` 返回：
- `train_cmd`：启动训练的命令
- `resume_cmd`：恢复训练的命令
- `run_dir`：检查点输出目录
- `total_epochs`：预期总训练轮次

### 测试与评估

```bash
# 在测试集上进行通用评估
python src/testing/test_general.py --device 0 --model-name MyModel --result-dir result

# 稳定性测试（模态丢失）
python src/testing/test_stability.py --device 0 --modal-dropout rgb --model-name RGB_Dropout_Test
python src/testing/test_stability.py --device 0 --modal-dropout ir --model-name IR_Dropout_Test

# 使用自定义权重路径
python src/testing/test_general.py --weights models/posttrain/MyModel/weights/best.pt --model-name MyModel
```

**测试参数（固定约定）：**
- `imgsz=640`，`rect=True`（保持 640×512 宽高比）
- `conf=0.25`，`iou=0.75`，`max_det=1000`
- 数据集：`src/cfg/datasets/dual_obb_dronevehicle.yaml`

### 数据预处理

```bash
# 预览裁剪后的数据及标签
python src/dataset_preprocess/verify_obb_preview.py --data-root data_croped --subset test --start-index 0

# 裁剪白边（已完成，仅作参考）
python src/dataset_preprocess/crop_white_borders.py --subset all --workers 8 --threshold 250
```

## 高层架构

### 模型配置系统

模型由 `src/cfg/model/` 中的 YAML 配置文件定义网络图。架构遵循以下流程：

```
YAML 配置 → parse_model()（位于 tasks.py）→ nn.Module 列表 → DetectionModel
```

**关键模块类别：**

| 模块类型 | 位置 | 用途 |
|---------|------|------|
| `IdentityInput` | `block.py:699` | 直通占位符，用于图分支起点 |
| `ModalitySelector` | `block.py:715` | 从 6 通道输入中提取 RGB（idx=1）或 IR（idx=2） |
| `FeatureAttentionConcat` | `fusion.py:65` | FA 融合：每模态 Inception+SE → 拼接 |
| `CrossModalFusionAttention` | `fusion.py:149` | CM-FA 融合：Inception + CrossModalSE → 拼接 |
| `CrossModalSE` | `fusion.py:105` | 跨模态联合 SE 注意力 |
| `Inception` | `fusion.py:7` | 4 分支多尺度特征提取 |
| `SEBlock` | `fusion.py:29` | 通道注意力（简化版 SE 模块） |

**双主干 YAML 结构模式：**

```yaml
backbone:
  - [-1, 1, IdentityInput, []]          # 层 0：入口点
  - [-1, 1, ModalitySelector, [1]]      # 层 1：提取 RGB
  - [-2, 1, ModalitySelector, [2]]      # 层 2：提取 IR（注意 -2 表示同一输入）
  # RGB 分支从层 1 开始
  - [1, 1, Conv, [64, 3, 2]]            # 层 3
  # ...（继续至 P3/P4/P5）
  # IR 分支从层 2 开始
  - [2, 1, Conv, [64, 3, 2]]            # 层 12
  # ...（继续至 P3/P4/P5）

head:
  # 在 P3/P4/P5 尺度使用 FeatureAttentionConcat 或 CrossModalFusionAttention 进行融合
  - [[7, 17], 1, FeatureAttentionConcat, []]   # RGB-P3（层 7）+ IR-P3（层 17）
  - [-1, 3, C2f, [256]]
  # ... P4、P5 类似
  - [[24, 26, 28], 1, OBB, [nc, 1]]     # 检测头
```

### 数据流

**训练阶段：**
1. `YOLODualDataset`（自定义，位于 `ultralytics-8.2/ultralytics/data/dataset.py`）加载 RGB+IR 图像对
2. 图像按通道维度拼接：`torch.cat([rgb, ir], dim=0)` → [6, H, W]
3. 主干中的 `ModalitySelector` 将输入分割为两个 3 通道流
4. 双主干网络独立处理每个模态
5. 融合模块（在 P3/P4/P5）结合特征
6. OBB 头输出旋转框预测

**关键实现细节：** 数据集加载器（`YOLODualDataset`）通过将路径中的 `img` 替换为 `imgr` 自动推断 IR 目录。因此数据集 YAML 只需指定 RGB 路径。

### 融合模块演进

三代融合模块：

1. **`FusionAttention`（fusion.py:48）**：Inception+SE 后的逐元素相加。**已弃用** - 导致信息瓶颈。

2. **`FeatureAttentionConcat`（fusion.py:65）**：使用通道拼接替代相加。每个模态有独立的 Inception+SE，输出拼接。用于 FA-Concat 模型。

3. **`CrossModalFusionAttention`（fusion.py:149）**：跨模态 SE 替代独立 SE。跨两个模态的联合全局池化允许基于质量的动态加权（如夜间抑制 RGB）。用于 CM-FA 模型。

### 训练功能（集成到框架中）

**模态丢失**（`drop_prob_rgb`、`drop_prob_ir`）：
- 训练期间随机将 RGB 或 IR 通道置零
- 互斥逻辑（从不同时丢失两个）防止全黑输入
- 最后轮次自动禁用（`close_dropout` 参数）
- 实现在 `ultralytics/models/yolo/detect/train.py`

**测试集作为验证集**（`use_test_as_val=True`）：
- 训练期间使用测试集进行验证

### 模型检查点组织

```
models/
├── formal/              # 完全训练的生产模型
│   ├── FA-Concat-FPN-PAN-neck/
│   └── CM-FA-Concat-FPN-PAN-neck/
├── posttrain/           # 迁移学习/微调模型
│   ├── FA-Concat_FPN-PAN_tuned/
│   └── CM-FA-Transferred/
└── baseline/            # 实验/训练中的模型
```

每个模型目录包含：
- `weights/best.pt` 和 `weights/last.pt`
- `args.yaml`（训练超参数）
- `results.csv`（每轮次指标）
- `confusion_matrix.png`（如已生成）

### 结果结构

测试输出保存至 `result/<model-name>/`：
- `<model-name>.json` 和 `<model-name>.csv`（汇总指标）
- `predictions.json`（来自验证器的原始预测）
- `confusion_matrix.png`（如由验证器生成）

## 关键文件关系

| 文件 | 用途 | 相关文件 |
|------|------|---------|
| `src/cfg/model/*.yaml` | 模型架构定义 | `ultralytics/nn/tasks.py`（parse_model） |
| `src/cfg/datasets/*.yaml` | 数据集路径和类别映射 | `ultralytics/data/dataset.py`（YOLODualDataset） |
| `ultralytics/nn/modules/fusion.py` | 自定义融合模块 | `ultralytics/nn/modules/__init__.py`（导出） |
| `ultralytics/nn/modules/block.py` | IdentityInput、ModalitySelector | 同上 |
| `src/trainning/*_train.py` | 训练入口点 | `train_manager.py`（编排） |
| `src/testing/test_general.py` | 标准评估 | `test_stability.py`（扩展） |

## 重要约定

**6 通道输入张量布局：**
- 通道 0-2：RGB
- 通道 3-5：IR（红外，复制为 3 通道以保持主干网络兼容性）

**OBB 标签格式：** `class cx cy w h angle`
- 除角度外所有值归一化到 [0, 1]
- 角度为弧度，有向包围框格式

**分辨率策略：**
- 训练：`imgsz=640`，`rect=False`（正方形，随机打乱）
- 验证/测试：`imgsz=640`，`rect=True`（矩形分桶，640×512）

**双目录约定：**
- `data/` - 带白边的原始图像
- `data_croped/` - 裁剪后的图像（用于训练）
- IR 目录通过将路径中的 `img` 替换为 `imgr` 自动推断

## 测试注意事项

- 评估始终使用 `data_croped/`（与训练一致）
- 稳定性测试使用 `StabilityValidator` 类，在预处理时注入模态丢失
- 测试比例 `< 1.0` 创建随机子集用于快速验证（通过种子保证确定性）


**注意**：谨记全局规则要求的“删除”文件方法
