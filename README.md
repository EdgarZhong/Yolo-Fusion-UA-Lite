# YOLO-Fusion-UA-Lite

**最终目标：** 构建一个在DroneVehicle数据集上表现优异的、轻量级的YOLO-based OBB任务双模态融合检测模型
**通过吸取最新论文成果，进行架构与方法创新，进一步提高轻量级模型的精度效果**

**技术栈：**

*   **环境：** mini-conda（在 PowerShell 先执行 `start-conda` 中的命令以启用） + Python 3.12
*   **框架：** PyTorch
*   **基础模型：** YOLOv8（固定使用本仓库随附的 `ultralytics-8.2` 源码）
*   **硬件：** 支持 CUDA 的 NVIDIA GPU（驱动版本需匹配 12.4 运行库）,目前使用RTX4060（8GB显存）
*   **核心数据集：** DroneVehicle

## **双模态有向目标检测模型开发与验证实践路径**

**核心思想：**  直接在`ultralytics-8.2/`目录中提供的官方框架源码上进行修改与扩展

**最终目标：**

1.  构建一个基于`ultralytics` YOLOv8-OBB原生功能的、可工作的双路输入基线模型。
2.  在该基线之上，实现并集成`FusionAttention`及其改进版本（FA-Concat）融合模块。
3.  通过在DroneVehicle数据集上的对比实验，量化验证改进模块的有效性。

### **阶段零：环境与数据准备**

1.  **环境配置：**
    *   创建并激活一个隔离的虚拟环境。
    *   通过标准包管理器安装所有必需的、不可修改的外部库依赖。

2.  **数据与标签准备：**
    *   获取原始数据集。
    *   执行一个一次性的**标签预处理脚本**。该脚本的核心逻辑必须完成：
        a. **输入：** 原始多模态标注。
        b. **几何统一：** 将所有原始的非标准几何标注（如四角点）在内存中转换为两种标准表示：一种用于鲁棒匹配（轴对齐），另一种用于精确描述（原始顶点）。
        c. **多模态合并：** 执行一个两阶段的智能合并算法。第一阶段使用低精度但快速的几何表示进行初步匹配；第二阶段对未匹配项，使用高精度的几何表示和更严格的匹配准则进行二次匹配。合并操作必须基于能保留所有几何信息的原始顶点。
        d. **最终格式化：** 将合并后的几何真值，通过最小面积包围算法，转换为目标检测框架所要求的最终参数化格式。
        e. **输出：** 生成一个与原始图像一一对应的、格式统一的最终标签集。

### **阶段一：构建并验证双路OBB基线框架**

**核心思想：** 直接在官方框架的基础上进行修改与扩展，而不是通过“猴子补丁”或自定义扩展的方式。

1. **在官方源码中编写自定义模块：**
   *   **数据模块：** 实现一个能够处理双路图像输入的数据处理类。
   *   **模型模块：** 实现一个能够程序化地、根据自定义配置文件构建双主干网络模型的类。该类应继承自官方框架的模型基类，并重写其前向传播逻辑以处理双路输入。
   
2. **定义基线模型配置：**
   *   在`src/cfg/datasets`创建一个自定义的数据集配置文件。该文件需描述双路输入的目录结构、通道顺序、标签格式等。
   *   在`src/cfg/model`创建一个自定义的配置文件。该文件需描述一个双主干网络结构，并在颈部使用最基础的拼接操作进行特征融合。此处请参考`yolo-fuse/`中的`Easy-level-Feature-Fusion.yaml`配置文件。

3. **编写主执行脚本：**

   *   创建一个主训练脚本在`src/trainning`中。

4. **训练与评估基线：**
   *   执行主训练脚本完成训练。
   *   编写并执行一个类似的验证脚本完成评估。
   *   记录所有实验配置和最终的性能指标，作为后续对比的基准。

### **阶段二：实现与验证高级融合模块 (FusionAttention & FA-Concat)**

此阶段已完成代码实现，FA-Concat模型正在训练中。实施方法详见`FusionAttention模块实施手册.md`。

**背景：** 
早期的 FusionAttention (FA-Add) 实验显示，虽然特征提取能力增强（收敛快），但由于 `Element-wise Add` 操作导致了严重的信息瓶颈，模型的 Recall 和最终 mAP 未能超越基线。

**当前策略：FA-Concat (无损增强版)**
1.  **改进逻辑：** 将 FA 模块的输出操作从“相加”改为“拼接 (Concat)”。
    *   **输入：** 双模态特征。
    *   **处理：** 经由 Inception (多尺度感知) 和 SE (去噪) 独立增强。
    *   **输出：** 将增强后的特征在通道维度拼接，无损保留所有信息。
2.  **实施步骤：**
    *   修改 `FusionAttention` 代码，实现 Concat 输出。
    *   更新模型配置文件，替换原有的简单 Concat 层。
    *   重新训练并对比基线。

### **阶段三：架构补全与动态感知迭代**

本阶段旨在解决当前架构缺失 FPN/PANet 以及缺乏跨模态动态感知的问题。

**迭代 v2.0: "The Completed Body" (架构补全版)**
*   **目标：** 解决当前模型最大的结构短板——缺失 Neck (FPN/PANet)。打通 P3/P4/P5 之间的特征流动，旨在提升 Precision 和 mAP。
*   **方案：** 在 FA-Concat 融合层之后，接入一个标准的 **YOLOv8 FPN + PANet** 结构。
*   **预期：** 小目标 Recall 和整体定位精度显著提升。
*   **实施指南：** 详见 `Neck_Construction_Guide.md`
*   模型正在训练中

**迭代 v2.1: "Cross-Modal Perception" (跨模态感知版)**
*   **目标：** 解决“不确定性感知”问题。
*   **方案：** 升级 FA 模块中的 SE 子模块为 **Cross-Modal SE**。让网络在计算权重时能同时看到 RGB 和 IR 的全局信息，从而实现动态的模态竞争（如在夜间自动抑制 RGB 权重）。
*   **配套：** 必须配合数据增强（裁白边 + Mosaic）进行训练。

**迭代 v2.2: "The Final Polish" (后训练微调)**
*   **目标：** 挖掘最强模型的最后潜力。
*   **方案：** 见双模态 YOLO-OBB 模型调参优化与策略分析报告。

### **阶段五：重启与鲁棒性优化 (Phase 5: Restart & Robustness Optimization)**

**背景：**
项目重启后，通过摸底测试发现现有最佳模型（M5）在红外（IR）模态丢失时，mAP 暴跌约 35%（0.778 -> 0.506）。虽然 DroneVehicle 数据集中 IR 图像质量普遍优于 RGB，模型依赖高质量模态属于合理行为，且 IR 缺失导致精度大幅下降符合预期；但这也反映出模型并未充分挖掘 RGB 模态的互补信息，仍有通过架构优化进一步提升精度的空间。
**修正**：经尝试，直接从 M5 模型迁移权重不可行。因此调整策略为模仿 M5 的训练流程，重新进行全量训练。

在此基础上，将“验证 CM-FA 模块的精度与稳定性优势”作为本阶段的具体验证目标，力求在实现极致精度的同时，兼顾模态缺失场景下的鲁棒性（双强互补，单强可用）。

**实施计划：**

1.  **基线摸底与归因分析：**
    *   使用 `test_stability.py` 对现有模型进行模态缺失测试，量化依赖程度。
    *   结论：RGB 特征提取能力未被有效激活，需强制模型学习 RGB 特征。

2.  **增强型模态随机失活 (Enhanced Modality Dropout)：**
    *   **策略：** 在训练过程中以 20% 概率随机丢弃 RGB 或 IR（互斥，防止全黑）。
    *   **目的：** 制造“模态缺失”的训练场景，迫使网络在单一模态下也能完成检测。
    *   **动态关闭：** 在训练的最后阶段（如最后 20 Epoch）自动关闭此增强，让模型在完整双模态数据上进行最终拟合。

3.  **全量重训练 (Full Retraining with COCO Transfer)：**
    *   **模型架构：** 采用 CM-FA-Concat（跨模态注意力），引入 Cross-Modal SE 模块实现动态权重分配。
    *   **权重初始化：加载 COCO 预训练的 YOLOv8n 权重**作为 Backbone 初始化（模仿 M5 的训练起点）。
    *   **训练策略：** 复用 M5 的成功经验，包括超参数、数据增强策略等。
    *   **泛化导向** ：在全量训练过程中，直接验证模型在整个测试集上的泛化能力。

4.  **最终验证：**
    *   训练结束后，再次运行稳定性测试，对比 mAP 下降幅度。

## 项目核心信息

### 进度情况
   **阶段一：已完成**（双主干简易基线与数据/训练流程搭建）
   **阶段二：已完成**（FusionAttention/FA‑Concat/CM‑FA‑Concat 模块实现与验证）
   **阶段三：已完成**（FPN/PAN 颈部结构补全与统一评估脚本）
   **阶段四：已完成**（最终微调：高分辨率探索与正则化难样本挖掘）
   **阶段五：进行中**（重启与鲁棒性优化：解决 IR 单模态依赖问题）
   当前状态：项目重启，专注于提升模型在模态缺失场景下的鲁棒性。统一测试标准：`imgsz=640`、`conf=0.25`、`iou=0.75`、`max_det=300`，并支持 `use_test_as_val=True` 在训练期以测试集做验证。

### 重要约定
- 任务类型：YOLO‑OBB（旋转框检测），标签为 6 列格式 `class cx cy w h angle`（归一化，角度为弧度），样例见 `data/train/trainlabels_yolo_obb/00001.txt`
- 输入模态与顺序约定：
  - 目录命名：`data/<subset>/<subset>img` 存放 RGB，`data/<subset>/<subset>imgr` 存放 IR，`data/<subset>/<subset>labels_yolo_obb` 存放适用于 OBB 任务的 6 列标签。
  - 模态顺序：始终为 “1: img(RGB), 2: imgr(IR)”；两个模态均为三通道输入。

- 分辨率策略 **（采用data_croped/下的裁切掉白边的数据集）**：训练阶段使用 `imgsz=640` 且 `rect=False`（开启随机打乱），验证与测试阶段使用 `imgsz=640` 且 `rect=True` 保持 640×512 的矩形分桶。
- 数据增强 **（采用data_croped/下的裁切掉白边的数据集）**：正式训练开启 `mosaic=1.0`，颜色增强保持关闭，保证IR图像数据分布符合真实情况；验证阶段不使用增强。
- 模态随机失活（Modality Dropout）：
  - **机制**：在训练过程中，以一定概率随机将 RGB 或 IR 通道的所有像素值置为 0，迫使模型学习不依赖单一模态的特征。
  - **策略**：
    - **互斥逻辑**：确保不会同时丢弃两个模态（防止输入全黑）。
    - **概率设定**：默认单模态丢弃概率为 0.2（即 20% 概率丢 RGB，20% 概率丢 IR，60% 概率保留双模态）。
    - **动态关闭**：在训练的最后阶段（如最后 10 Epoch）自动关闭此增强，让模型在完整双模态数据上进行最终拟合。
  - **用法**：已完全集成至 `ultralytics` 框架源码 (`models/yolo/detect/train.py`)，直接通过训练参数控制，不再依赖外部 Hook 脚本，实现了训练脚本与框架功能的解耦。
- **新增训练参数**（直接在 `model.train()` 中传递）：
  - `use_test_as_val=True`：在训练期间直接使用测试集（test）作为验证集，以获得更真实的泛化性能反馈。
  - `add_val_to_train=True`：当启用 `use_test_as_val` 时，脚本会自动将原验证集（val）合并到训练集（train）中参与训练，最大化利用数据（17789+1445 -> 19234 样本）。
  - `drop_prob_rgb=0.2`：RGB 模态随机失活概率。
  - `drop_prob_ir=0.2`：IR 模态随机失活概率。
  - `close_dropout=10`：在训练结束前 N 个 Epoch 关闭模态失活增强，确保最终收敛稳定性。
- **测试参数统一**
  - `imgsz=640`、`conf=0.25`、`iou=0.75`、`max_det=300`

### 环境与激活

- 初始化 Conda（PowerShell）：
  - `& "C:\DevLib\miniconda3\Scripts\conda.exe" shell.powershell hook | Out-String | Invoke-Expression`
- 激活项目环境：
  - `conda activate .\\.conda\\ultra82-py312`
- 版本锁定：`ultralytics==8.2.103`、`numpy==1.26.4`、`torch==2.6.0+cu124`（CUDA 12.4）

### 数据与目录结构（裁切后）

- 原始图像目录（统一约定）：
  - `data/train/trainimg/`（RGB） 与 `data/train/trainimgr/`（IR）
  - `data/val/valimg/`（RGB）   与 `data/val/valimgr/`（IR）
  - `data/test/testimg/`（RGB） 与 `data/test/testimgr/`（IR）
- 裁切后图像目录（用于训练/验证）：
  - `data_croped/train/trainimg/`（RGB） 与 `data_croped/train/trainimgr/`（IR）
  - `data_croped/val/valimg/`（RGB）   与 `data_croped/val/valimgr/`（IR）
  - `data_croped/test/testimg/`（RGB） 与 `data_croped/test/testimgr/`（IR）
- 标签目录：
  - `data_<或>data_croped/<subset>/{subset}labels_yolo_obb/`（与图像同名 `.txt`）
- 目录快照：
  - `data/` 内含 `classes.txt`、`mismatch_obb.txt` 与 `数据集预处理逻辑.md`
  - 预处理脚本：`src/dataset_preprocess/preprocess_obb.py`、`verify_obb_preview.py`、`clean_mismatch.py`、`crop_white_borders.py`
- 数据集预处理（白边裁切）已完成；`crop_meta.json` 显示统一裁切为 `x100_y100_w640_h512`
- **已保证此描述实际准确，无需怀疑**
- **修改`ultralytics-8.2/`目录中源码的标签映射和数据集加载逻辑以适配约定--此项已完成**

### 训练与验证（更新）

 - **目录约定**
 - 快速验证脚本：如果进行全量训练，需要编写冒烟测试脚本，确保模型在训练过程中能正常运行。如 `src/trainning/fa_concat_fpn_pan_quick_train.py`（fraction=0.05）。
 - **数据集 YAML 已切换至裁切后目录：** `src/cfg/datasets/dual_obb_dronevehicle.yaml` 的 `train/val/test` 指向 `data_croped/<subset>/<subset>img`
 - 验证脚本统一为 `imgsz=640` 与 `rect=True`，保持 640×512 输入：`src/testing/test_general.py:51`

### 测试脚本：`src/testing/test_general.py` 用法

- 原理说明（按 log 校正）：脚本使用 `OBBValidator` 在测试集进行评估，固定 `imgsz=640`、`rect=True`，透传 `max_det=1000`（`src/testing/test_general.py:55,97-115,138-141`）。单双模态的训练与验证流程保持解耦，参数统一由数据集 YAML 与脚本内路径宏控制。
- 推荐做法：直接修改脚本顶部路径宏以匹配你的模型与数据路径（`src/testing/test_general.py:46-55`）。
- 参数列表：
  - `--device cpu|<gpu_index>` 运行设备，默认 `0`（GPU）。
  - `--weights <pt或目录>` 权重文件或训练输出目录（未提供时按脚本宏自动查找，见 `_find_weights`：`src/testing/test_general.py:58-72`）。
  - `--model-name <str>` 结果目录名（用于 `result/<model-name>/`）。
  - `--result-dir <path>` 结果输出根目录（默认 `result`）。
  - `--test-ratio <0-1]` 测试集比例；小于 1.0 时按比例构建子集 DataLoader（`src/testing/test_general.py:117-136`）。
  - `--test-aug` 测试增强（多尺度/翻转）。
  - `--iou <float>` NMS 的 IOU 阈值（默认 `0.75`）。
- 输出文件：
  - `result/<model-name>/<model-name>.json` 与 `result/<model-name>/<model-name>.csv`（写入逻辑见 `src/testing/test_general.py:174-193`）。
  - 验证器生成 `result/<model-name>/predictions.json`（按 Ultralytics 约定）。
- 示例命令（建议先按需修改脚本顶部路径宏）：
  - `python src/testing/test_general.py --device 0 --model-name Final_Recall_640_Regularized --result-dir result`
  - `python src/testing/test_general.py --weights models/posttrain/Final_Recall_640_Regularized --model-name RecallReg --result-dir result`
  - `python src/testing/test_general.py --weights models/IR-YOLOv8n/from_scrach/train/weights/best.pt --model-name IR-YOLOv8n --device 0`
  - `python src/testing/test_general.py --device cpu --test-ratio 0.2 --model-name quick-sample`


### 模型信息表（按名称序列汇总）

| 名称 | 权重存放目录 | 测试结果存放目录 | 使用的数据集 | 测试集mAP |
| :-- | :-- | :-- | :-- | :-- |
| 双主干简单拼接基线模型 | `models/formal/dualbackbone-easy-obb-formal6/weights` | `result/baseline-100epoch` | 有白边原图 | 0.6632 |
| IR单模态基线模型 | `models/IR-YOLOv8n/from_scrach/train/weights` | `result/IR-YOLOv8n` | 有白边原图 | 0.7402 |
| 原版FusionAttention融合模型 | `models/formal/fusion-attention/dualbackbone-fusionattention-obb/weights` | `result/fusionattention-only-120epoch` | 有白边原图 | 0.6242 |
| FeatureAttentionConcat（FA-Concat）型融合模型 | `models/formal/dualbackbone-FA-Concat-obb/weights` | `result/dualbackbone-FA-Concat-100epoch` | 有白边原图 | 0.6647 |
| 完整YOLO neck结构的FA-Concat模型 | `models/formal/FA-Concat-FPN-PAN-neck/weights` | `result/FA-Concat-FPN-PAN-neck-100epoch` | 裁切白边 | 0.7297 |
| 跨模态注意力融合的CM-FA-Concat模型 | `models/formal/CM-FA-Concat-FPN-PAN-neck/weights` | `result/CM-FA_FPN-PAN_neck` | 裁切白边 | 0.7177 |
| 主干权重迁移训练的FA-Concat_FPN-PAN_tuned | `models/posttrain/FA-Concat_FPN-PAN_tuned/weights` | `result/FA-Concat_FPN-PAN_tuned` | 裁切白边 | 0.7617 |
| 高分辨率微调模型（基于FA-Concat_FPN-PAN_tuned） | `models/posttrain/Final_Polish_800_FlipScale/weights` | `result/Final_Polish_800_FlipScale` | 裁切白边 | 0.7445 |
| 正则化与难样本挖掘微调模型（基于FA-Concat_FPN-PAN_tuned） | `models/posttrain/Final_Recall_640_Regularized/weights` | `result/Final_Recall_640_Regularized` | 裁切白边 | 0.7456 |

### 多模型结果汇总（7模型×6维）

| 模型代号 | 模型名称 | mAP50 | mAP95 | Recall | Precision | FPS | 收敛效率 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| M0 | IR-YOLOv8n | 0.66190 | 0.50329 | 0.62199 | 0.67828 | 166.45462 |  4.34783 |
| M1 | Dual-Easy-Concat | 0.66325 | 0.53292 | 0.63298 | 0.66060 | 127.74533 |  3.12500 |
| M2 | Dual-FA-Concat(without neck) | 0.66469 | 0.53631 | 0.64618 | 0.66124 | 121.81166 | 4.16667 |
| M3 | FA-Concat (Scratch) | 0.72973 | 0.58864 | 0.67730 | 0.74700 | 117.89529 | 7.14286 |
| M4 | CM-FA (Scratch) | 0.71774 | 0.57613 | 0.67340 | 0.73448 | 118.41942 | 7.69231 |
| M5 | FA-Concat (Tuned) | 0.76170 | 0.61536 | 0.70353 | 0.78526 | 114.69225 | 12.50000 |
| M6 | FA-Concat (Reg) | 0.74563 | 0.60353 | 0.70054 | 0.76284 | 109.52974 | 100.00000 |

> 数据来源：`result/多模型结果汇总.csv`（由 `src/tools/build_summary_table.py` 自动生成），速度基准由 `src/testing/benchmark_speed_subset.py` 测得。

**白边裁切与预览**

 - 白边裁切工具：`src/dataset_preprocess/crop_white_borders.py`
   - 命令（全量处理并写出）：`python src/dataset_preprocess/crop_white_borders.py --subset all --workers 8 --threshold 250`
   - 输出：`data_croped/<subset>/<subset>img|imgr|labels_yolo_obb/`，统计 `data_croped/crop_meta.json`
 - 预览裁切/非裁切数据与 OBB 标签：`src/dataset_preprocess/verify_obb_preview.py`（显示类名）
   - 命令（裁切集预览）：`python src/dataset_preprocess/verify_obb_preview.py --data-root data_croped --subset test --start-index 0`
   - 支持参数：`--data-root` 切换 `data/` 与 `data_croped/`；`--subset` 选择子集；`--start-index` 指定起始样本


## 常见问题与说明
- 若验证阶段出现“Mean of empty slice / invalid value encountered in divide”：
  - 检查是否使用 HBB 检测任务读取 OBB 标签或标签路径不匹配；请确保使用 OBB 任务与 `labels_yolo_obb` 目录
- 若出现 pin_memory 线程异常：
  - 将 `workers` 降至 0/2，保证缓存与并发稳定
- 若日志显示输入为正方形：
  - 训练阶段为 `imgsz=832` 且 `rect=False`（批次随机打乱）；验证/测试阶段为 `imgsz=832` 且 `rect=True`（按长宽比分桶，自动关闭 shuffle）。不再使用矩形尺寸对（如 `[704,832]`），以避免任何隐性裁切与标签调整。

## 参考与附属记录
- Conda 激活命令位于：`start-conda`
- `ultralytics-8.2/` 目录下的docs包含可以参阅的文档
- 各进度阶段的文档记录见下方表格


### 环境搭建（基于 ultralytics-8.2，推荐稳定路径）

- 初始化 Conda（PowerShell）：
  - `& "C:\DevLib\miniconda3\Scripts\conda.exe" shell.powershell hook | Out-String | Invoke-Expression`
- 创建并激活独立环境（Python 3.12）：
  - `conda create -y -p .\.conda\ultra82-py312 python=3.12`
  - `conda activate .\.conda\ultra82-py312`
- 安装依赖（固定在与 8.2 兼容的版本区间）：
  - `python -m pip install -q numpy==1.26.4 matplotlib opencv-python pillow pyyaml requests scipy tqdm psutil py-cpuinfo pandas seaborn ultralytics-thop`
  - `python -m pip install -q --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124`
- 验证 GPU：
  - `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
- 在 `ultralytics-8.2/` 目录中以可编辑模式安装：
  - `python -m pip install -e .`

**固定版本说明（针对 ultralytics-8.2）**

- `numpy < 2.0.0`（建议 `1.26.4`）以避免 CLI 版本校验失败
- `torch/vision` 对应 `cu124` 发行包；无需系统 CUDA Toolkit（`nvcc`）
- 驱动要求：`nvidia-smi` 的 `CUDA Version` 建议不低于 12.4

### 数据集清洗与规模变更

- 清洗依据：数据集预处理脚本在第二/第三阶段 OBB 匹配中发现跨模态类别不一致时，会记录到 `data/mismatch_obb.txt`
- 清洗脚本：`python clean_mismatch.py data` 删除记录样本的两侧图像、两份原 XML 与对应 OBB 标签
- 删除前后数据组数量（每组一个样本对）：
  - train：17990 → 17789
  - val：1469 → 1445
  - test：8980 → 8876
  - 总计剔除：329 组

### 废除项与最新实践

- 丢弃：直接依赖 `site-packages` 中的高版本 `ultralytics (8.3.x)`，以及在训练阶段对库进行猴子补丁的方案。
- 最新实践：将 `ultralytics-8.2` 作为稳定基础，按上述步骤在独立环境中以可编辑模式安装；可以直接修改本地源码，确保可控与可复现。
- 谨记：实际使用的框架源码位于：`ultralytics-8.2/ultralytics`。

### 项目目录结构（目录树）

```
YOLO-Fusion-UA-Lite/
├─ data/
│  ├─ train/ val/ test/
│  ├─ classes.txt
│  ├─ mismatch_obb.txt
│  └─ 数据集预处理逻辑.md
├─ data_croped/
│  └─ .gitkeep
├─ models/
│  ├─ IR-YOLOv8n/
│  │  └─ from_scrach/train/weights/
│  ├─ baseline/
│  ├─ formal/
│  │  ├─ dualbackbone-easy-obb-formal6/
│  │  ├─ dualbackbone-FA-Concat-obb/
│  │  ├─ FA-Concat-FPN-PAN-neck/
│  │  └─ CM-FA-Concat-FPN-PAN-neck/
│  ├─ fusion-attention/
│  ├─ migratory/
│  └─ posttrain/
│     ├─ FA-Concat_FPN-PAN_tuned/
│     ├─ Final_Polish_800_FlipScale/
│     ├─ Final_Recall_640_Regularized/
│     └─ Final_Recall_640_Regularized_val_on_test/
├─ result/
│  ├─ result_modal_dropout/ … [新增] 模态随机失活稳定性摸底结果
│  ├─ baseline-100epoch/ … predictions.json
│  ├─ IR-YOLOv8n/ … 指标图片/CSV/JSON
│  ├─ dualbackbone-FA-Concat-100epoch/ …
│  ├─ FA-Concat-FPN-PAN-neck-100epoch/ …
│  ├─ CM-FA_FPN-PAN_neck/ …
│  ├─ FA-Concat_FPN-PAN_tuned/ … confusion_matrix*.png
│  ├─ Final_Polish_800_FlipScale/ …
│  ├─ Final_Recall_640_Regularized/ …
│  ├─ benchmark_speed_subset.csv
│  ├─ 多模型结果汇总.csv
│  ├─ 训练曲线.png
│  └─ radar_chart_final.png
├─ src/
│  ├─ cfg/
│  │  ├─ datasets/dual_obb_dronevehicle.yaml 等
│  │  └─ model/dualbackbone_easy_obb.yaml 等
│  ├─ dataset_preprocess/
│  ├─ trainning/
│  ├─ testing/
│  └─ tools/
├─ ultralytics-8.2/
│  ├─ ultralytics/ …（框架源码）
│  ├─ docs/
│  ├─ examples/
│  └─ README.zh-CN.md
├─ docs/
│  └─ ... (文档)
├─ start-conda
└─ .gitignore
```

### 文档目录（按时间先后）

- [data/数据集预处理逻辑.md](data/数据集预处理逻辑.md)：数据清洗与 OBB 标签统一流程，含双阶段几何匹配与白边裁切规范（裁切元数据 `crop_meta.json`）。
- [val_memory_analysis.md](docs/val_memory_analysis.md)：验证集显存爆炸原因分析，最终改用 CPU 计算 NMS 的 IOU。
- [FusionAttention模块实施手册.md](docs/FusionAttention模块实施手册.md)：FusionAttention 的设计与实现，FA‑Concat/CM‑FA‑Concat 的升级路径与训练/评估建议。
- [cuDNN_STATUS_EXECUTION_FAILED_训练错误排查与解决.md](docs/cuDNN_STATUS_EXECUTION_FAILED_训练错误排查与解决.md)：训练时 cuDNN 执行失败的成因与修复路径，含稳定性调试建议。
- [Neck_Construction_Guide.md](docs/Neck_Construction_Guide.md)：完整 YOLOv8 FPN+PAN 颈部结构的插入方法与 YAML 配置指南。
- [FA-Concat-neck权重迁移训练和深度调优手册.md](docs/FA-Concat-neck权重迁移训练和深度调优手册.md)：COCO 预训练权重的双主干映射与热启动训练策略，含 Dropout/Mosaic 课程控制与超参表。
- [FA-Concat_neck_tuned最终高分辨率微调.md](docs/FA-Concat_neck_tuned最终高分辨率微调.md)：高分辨率（800）微调实施手册与参数设定、预期与验证注意事项。
- [提升recall的最终微调.md](docs/提升recall的最终微调.md)：正则化与损失重加权的难样本挖掘策略，总结统一测试标准与实施要点。
- [双模态 YOLO-OBB 模型调参优化与策略分析报.md](docs/双模态 YOLO-OBB 模型调参优化与策略分析报.md)：两阶段微调（高分辨率/正则化）结果与失败归因分析、后续改进方向建议。
- [多模型性能评估与可视化实施文档.md](docs/多模型性能评估与可视化实施文档.md)：最终 7 模型 × 6 维指标评估方案，含速度基准脚本 `src/testing/benchmark_speed_subset.py`、数据汇总 `result/多模型结果汇总.csv` 与雷达图脚本 `src/tools/plot_radar_final.py`。
- [CM-FA 模块迁移训练与稳定性验证实施方案](docs/CM-FA_Transfer_and_Stability_Plan.md)：**[NEW]** 从最佳模型迁移训练 CM-FA 模块与稳定性验证实施方案。

