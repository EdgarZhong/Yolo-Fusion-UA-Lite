# CM-FA 模块迁移训练与稳定性验证实施方案 (细化版)

## 1. 背景与目标
本项目旨在进一步验证 Cross-Modal Feature Attention (CM-FA) 模块在复杂场景下的有效性。
当前阶段目标是从现有的最佳模型 **FA-Concat_FPN-PAN_tuned (M5)** 出发，通过迁移训练的方式，快速构建并验证 CM-FA 改进模型。
同时，为了验证新模型在极端条件下的鲁棒性，将改进测试器以支持“模态随机失活”功能，并进行稳定性对比测试。

## 2. 迁移训练实施细节

### 2.1 源模型与目标模型
*   **源模型 (Source)**: `FA-Concat_FPN-PAN_tuned`
    *   权重路径: `models/posttrain/FA-Concat_FPN-PAN_tuned/weights/best.pt`
    *   融合模块: `FeatureAttentionConcat` (包含 `inc_rgb`, `inc_ir`, `se_rgb`, `se_ir`)
*   **目标模型 (Target)**: `CM-FA-Transferred`
    *   配置文件: `src/cfg/model/CM-FA_Concat_FPN-PAN_neck.yaml`
    *   融合模块: `CrossModalFusionAttention` (包含 `inc_rgb`, `inc_ir`, `cm_se`)

### 2.2 权重迁移策略 (Layer-wise Transfer Strategy)
我们将构建一个专用的权重迁移逻辑，在加载时执行以下映射：

| 模块区域 | 源模型组件 | 目标模型组件 | 迁移操作 |
| :--- | :--- | :--- | :--- |
| **Backbone** | `backbone.*` | `backbone.*` | ✅ **完全迁移** (保留强大的特征提取能力) |
| **Neck** | `neck.*` (FPN/PAN) | `neck.*` | ✅ **完全迁移** (保留多尺度特征聚合能力) |
| **Head** | `head.*` (Detect) | `head.*` | ✅ **完全迁移** (保留检测与分类能力) |
| **Fusion (P3/P4/P5)** | `inc_rgb`, `inc_ir` | `inc_rgb`, `inc_ir` | ✅ **完全迁移** (保留 Inception 特征提取能力) |
| **Fusion (P3/P4/P5)** | `se_rgb`, `se_ir` | (无) | ❌ **丢弃** (结构不兼容) |
| **Fusion (P3/P4/P5)** | (无) | `cm_se` | 🆕 **随机初始化** (需重新训练) |

### 2.3 训练参数策略

*   **脚本**: 新建 `src/trainning/cm_fa_transfer_train.py`
*   **总轮次 (Epochs)**: 30 (修正：原计划 50，考虑到微调性质缩短)
*   **学习率 (LR)**: `lr0=0.002` (从头训练的 1/5，避免破坏预训练权重)
*   **优化器**: SGD
*   **冻结策略 (Freeze Schedule)**:
    *   **Phase 1 (Epoch 0-9)**: **冻结** Backbone, Neck, Head, Fusion.Inception。**仅训练** `cm_se` 模块。
    *   **Phase 2 (Epoch 10-29)**: **解冻全网**。
*   **数据增强**: 保持与源模型一致 (`mosaic=1.0`, `mixup=0.0`, 开启裁切白边数据)。
*   **模态随机失活**: 开启 (`prob=0.2` 增强版)，最后 5 Epoch 关闭。

## 3. 稳定性验证方案 (Stability Verification)

### 3.1 测试工具改造
新建/修改测试脚本 `src/testing/test_stability.py`，增加参数 `--modal-dropout [mode]`：
*   `none`: 标准测试 (基准)。
*   `rgb`: 将 RGB 输入全置为 0 (模拟夜间/摄像头失效)。
*   `ir`: 将 IR 输入全置为 0 (模拟热成像失效)。
*   `random`: 对每个样本随机置零某一模态 (模拟不稳定传输)。

### 3.2 对比实验设计
在 `result_modal_dropout/` 下生成对比报告：
1.  **Baseline**: `FA-Concat` (Source Model)
    *   测试其在 RGB/IR 缺失下的 mAP 衰减幅度。
    *   预期：由于是独立 SE，单模态缺失可能导致该路特征全是噪声，直接拼接后可能干扰检测头。
2.  **Ours**: `CM-FA` (Transferred Model)
    *   测试其在 RGB/IR 缺失下的 mAP 衰减幅度。
    *   预期：Cross-Modal SE 应能感知到某一路能量极低（失效），从而自动降低该路权重，减少噪声干扰，衰减幅度应显著小于 Baseline。

## 4. 执行行动路径
1.  **基线摸底**: 使用改造后的测试器，先测 `FA-Concat` 的稳定性 (作为对照组)。
2.  **代码实现**: 完成 `cm_fa_transfer_train.py` 和 `test_stability.py`。
3.  **实施训练**: 启动迁移训练。
4.  **最终验证**: 训练完成后，测试 `CM-FA` 的稳定性并对比。
