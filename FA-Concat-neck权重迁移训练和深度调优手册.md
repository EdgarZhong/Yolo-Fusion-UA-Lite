# 双模态 YOLO-OBB 权重迁移与深度调优实施手册

## 1. 项目背景与目标
本项目旨在构建高性能的 DroneVehicle 双模态（RGB+IR）目标检测模型。当前的架构为 **FA-Concat + Neck**（双主干 + FeatureAttentionConcat 融合 + FPN/PAN 颈部）。

为了解决双主干模型因随机初始化导致的收敛缓慢及特征提取能力不足的问题，本方案定义了一套标准化的**权重热启动（Warm Start）**与**深度微调**流程。该流程通过将官方 YOLOv8n 的预训练参数精准迁移至双模态骨干，并配合“模态 Dropout”等鲁棒性增强策略，以实现模型性能的最大化。

## 2. 系统路径与命名规范

为确保实验的可复现性与文件管理的规范性，请严格遵循以下路径定义：

| 资产类型 | 路径/名称约定 | 说明 |
| :--- | :--- | :--- |
| **源模型配置** | `src/cfg/model/dualbackbone_FA-Concat.yaml` | 待训练的双主干网络结构定义 |
| **源权重文件** | `yolov8n.pt` | 请确保项目根目录下存在此官方权重 |
| **迁移工具脚本** | `tools/transfer_dual_weights.py` | 用于执行权重复制的独立脚本（需新建） |
| **热启动权重** | `models/migratory/dualbackbone_FA_Concat_pretrained.pt` | 迁移工具生成的中间产物 |
| **训练执行脚本** | `src/trainning/FA-Concat-neck_post_train.py` | 包含特定增强逻辑的训练脚本（需新建） |
| **最终输出目录** | `models/posttrain/FA-Concat_FPN-PAN_tuned/` | 训练日志与最终权重的保存位置 |
| **最终模型名称** | `FA-Concat_FPN-PAN_tuned` | 实验代号 |

---

## 3. 阶段一：权重迁移实施 (Weight Transfer)

**目标**：将单模态的 COCO 预训练能力“克隆”到双模态模型的两个独立分支中。

### 3.1 工具脚本开发 (`src/tools/transfer_dual_weights.py`)
开发人员需编写独立脚本，不依赖训练器，直接操作 PyTorch 的 `state_dict`。

### 3.2 映射逻辑定义
官方 `yolov8n` 主干包含 10 层（索引 0-9）。在我们的 `dualbackbone` 配置中，Layer 0 为 `IdentityInput` 占位层，因此存在索引偏移。请执行以下**深拷贝（Deep Copy）**操作：

1.  **RGB 分支映射**：
    *   源：官方权重 `model.0` 至 `model.9`
    *   目标：双主干权重 `model.1` 至 `model.10`
    *   *操作：逐层、逐张量复制参数（Weights & Biases）。*

2.  **IR 分支映射**：
    *   源：官方权重 `model.0` 至 `model.9`
    *   目标：双主干权重 `model.11` 至 `model.20`
    *   *操作：逐层、逐张量复制参数。*

### 3.3 输出
执行脚本后，应验证 `models/migratory/` 目录下生成了 `dualbackbone_FA_Concat_pretrained.pt`，且加载时无形状不匹配报错。

---

## 4. 阶段二：深度调优训练策略

**目标**：基于热启动权重，通过特定的增强策略与超参控制，完成特征融合层的训练与全局微调。

### 4.1 训练脚本开发 (`src/trainning/FA-Concat-neck_post_train.py`)
该脚本需继承标准训练流程，但必须注入以下两个核心的**生命周期控制逻辑**：

#### A. 模态随机 Dropout (Modality Dropout)
*   **功能描述**：在数据加载或变换阶段，随机丢弃某一模态的输入信息，强迫融合层处理单模态缺失的情况。
*   **实施逻辑**：
    *   对每一个 Batch 中的样本：
        *   以 **10%** 独立概率将 RGB 图像全像素置零（Black Frame）。
        *   以 **10%** 独立概率将 IR 图像全像素置零（Black Frame）。
*   **生命周期**：**仅在训练的前 175 个 Epoch 生效**。在最后 25 个 Epoch 必须自动关闭，以确保模型在完整的双模态分布上收敛。

#### B. Mosaic 增强控制
*   **实施逻辑**：设置 `close_mosaic=25`。
*   **协同作用**：确保在最后 25 个 Epoch，`Mosaic` 拼接增强与 `Modality Dropout` 同时关闭，模型进入“高清原图+全模态输入”的最终微调阶段。

---

## 5. 关键超参设置表 (Hyperparameters)

在训练脚本中，请通过参数覆盖确保以下设置生效。这些设置是针对 DroneVehicle 数据集特性（小目标、夜间 RGB 噪点高、红外物理属性强）的针对性优化。

| 参数项 | 设定值 | 核心依据 |
| :--- | :--- | :--- |
| **model** | `models/migratory/dualbackbone_FA_Concat_pretrained.pt` | **必须**加载阶段一生成的迁移权重。 |
| **epochs** | `200` | 给予融合层充分的学习时间，补偿 Dropout 带来的收敛难度。 |
| **freeze** | `10` | **冻结前 10 轮 Backbone**。防止随机初始化的 Neck 产生的剧烈梯度破坏预训练的主干特征。 |
| **imgsz** | `640` | 平衡训练效率与显存占用。 |
| **close_mosaic**| `25` | 最后 25 轮在真实大图分布上微调，同步关闭模态 Dropout。 |
| **hsv_h** | `0.0` | **关闭色调增强**。严禁破坏 IR 的物理温度属性及 RGB 的夜间特征。 |
| **hsv_s** | `0.0` | **关闭饱和度增强**。同上。 |
| **hsv_v** | `0.0` | **关闭亮度增强**。同上。 |
| **degrees** | `0.0` | **关闭旋转增强**。避免旋转框插值带来的 OBB 标签精度损失。 |
| **translate** | `0.1` | 保持适度的平移几何增强。 |
| **scale** | `0.5` | 保持适度的尺度几何增强。 |
| **fliplr** | `0.5` | 开启左右翻转。 |
| **project** | `models/posttrain` | 指定输出根目录。 |
| **name** | `FA-Concat_FPN-PAN_tuned` | 指定实验名称。 |

---

## 6. 开发者实施核查清单 (Checklist)

在启动训练前，请逐项确认：

1.  [ ] **权重就位**：`models/migratory/` 下已生成迁移后的 `.pt` 文件，且文件大小正常。
2.  [ ] **Dropout 植入**：代码中已包含模态置零逻辑，且该逻辑受 `current_epoch < (total_epochs - 25)` 条件约束。
3.  [ ] **增强互斥**：确认配置中 `HSV` 系列参数与 `Degrees` 参数已显式设为 `0.0`。
4.  [ ] **冻结策略**：训练启动后的首屏日志中，明确显示冻结了骨干层（例如 `Freezing layer...`）。
5.  [ ] **输出路径**：确认最终模型将保存至 `models/posttrain/FA-Concat_FPN-PAN_tuned/`。