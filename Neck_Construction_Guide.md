# YOLOv8 Neck 构建与配置指南

本指南旨在指导如何在现有的双主干融合模型（FA-Concat + C2f）基础上，恢复标准的 **YOLOv8 FPN + PANet** 结构，以解决特征层级割裂问题，提升小目标检测能力和定位精度。

## 1. 架构逻辑图解

当前模型在“融合”之后直接连接了检测头。我们需要在中间插入一个完整的 Neck。

**数据流向：**

1.  **入口融合 (Existing):**
    *   RGB/IR 主干特征 -> FA-Concat 模块 -> C2f 模块 -> **Fused_P3, Fused_P4, Fused_P5**
    *   *状态：* 得到了三个尺度的单流特征，但这三个特征之间还没有进行交互。

2.  **FPN (Feature Pyramid Network - 新增):**
    *   **Top-down 路径:** 将高层语义（P5）传给低层（P3）。
    *   `Fused_P5` (上采样) + `Fused_P4` -> `Neck_P4`
    *   `Neck_P4` (上采样) + `Fused_P3` -> `Neck_P3` (此层拥有最强的语义增强的小目标特征)

3.  **PANet (Path Aggregation Network - 新增):**
    *   **Bottom-up 路径:** 将低层定位信息（P3）传回高层（P5）。
    *   `Neck_P3` (下采样) + `Neck_P4` -> `Final_P4`
    *   `Final_P4` (下采样) + `Fused_P5` (或Neck_P5) -> `Final_P5`

4.  **检测头 (Head):**
    *   接收 `Neck_P3`, `Final_P4`, `Final_P5` 进行预测。

## 2. 配置文件编写指南 (YAML)

请在你的模型配置文件（`src/cfg/model/`下）中，找到 `head` 部分，按以下步骤修改。

### 步骤 A: 确认入口融合层的索引

首先，你需要明确你的三个融合特征层（Fused_P3/P4/P5）在网络中的层级索引（Index）。
假设你的配置文件中，融合部分的写法如下：

```yaml
# ... (Backbone) ...

# [Fused_P3] - 假设这是第 24 层
- [[7, 17], 1, FeatureAttentionConcat, [64]] 
- [-1, 1, C2f, [64]]  # 24: Fused_P3

# [Fused_P4] - 假设这是第 26 层
- [[9, 19], 1, FeatureAttentionConcat, [128]]
- [-1, 1, C2f, [128]] # 26: Fused_P4

# [Fused_P5] - 假设这是第 28 层
- [[12, 22], 1, FeatureAttentionConcat, [256]]
- [-1, 1, C2f, [256]] # 28: Fused_P5
```

*注意：请务必根据你实际的 YAML 文件计算层号，或者使用 `model.info()` 打印出的结构来确认索引。以下示例假设索引为 24, 26, 28。*

### 步骤 B: 插入 FPN + PANet 结构

在上述代码之后，添加以下标准 Neck 结构：

```yaml
# === 颈部 (Neck): 标准 FPN + PANet ===

# --- FPN (自顶向下) ---
# 29: 上采样 Fused_P5 (来自 layer 28)
- [-1, 1, nn.Upsample, [None, 2, 'nearest']] 

# 30: 拼接 (上采样的P5 + Fused_P4) -> 注意这里的 26 是 Fused_P4 的索引
- [[-1, 26], 1, Concat, [1]]                 

# 31: 融合得到 Neck_P4
- [-1, 3, C2f, [128]]                        

# 32: 上采样 Neck_P4
- [-1, 1, nn.Upsample, [None, 2, 'nearest']] 

# 33: 拼接 (上采样的P4 + Fused_P3) -> 注意这里的 24 是 Fused_P3 的索引
- [[-1, 24], 1, Concat, [1]]                 

# 34: 融合得到 Neck_P3 (小目标特征极大增强)
- [-1, 3, C2f, [64]]                         

# --- PANet (自底向上) ---
# 35: 下采样 Neck_P3
- [-1, 1, Conv, [64, 3, 2]]                  

# 36: 拼接 (下采样的P3 + Neck_P4) -> 注意这里的 31 是 Neck_P4 的索引
- [[-1, 31], 1, Concat, [1]]                 

# 37: 融合得到 Final_P4
- [-1, 3, C2f, [128]]                        

# 38: 下采样 Final_P4
- [-1, 1, Conv, [128, 3, 2]]                 

# 39: 拼接 (下采样的P4 + Fused_P5) -> 注意这里的 28 是 Fused_P5 的索引
- [[-1, 28], 1, Concat, [1]]                 

# 40: 融合得到 Final_P5
- [-1, 3, C2f, [256]]                        
```

### 步骤 C: 连接检测头

最后，修改检测头的输入源，使其指向 Neck 的输出层，而不是之前的 Fused 层。

```yaml
# === 头部 (Head) ===
# 输入变为: Neck_P3 (34), Final_P4 (37), Final_P5 (40)
- [[34, 37, 40], 1, OBB, [nc, 1]] 
```

## 3. 验证与调试

1.  **检查报错：** 运行训练脚本时，如果报 `IndexError` 或通道数不匹配错误，通常是因为 `Concat` 的 `from` 索引填错了。请仔细核对层号。
2.  **结构确认：** 使用 `model.info()` 打印模型结构，确认 Neck 部分的层数和连接关系符合预期（例如 Upsample 和 Concat 是否正确连接到了目标层）。
3.  **预期参数量：** 相比于无 Neck 版本，参数量应增加约 1M 左右。

完成此构建后，你的模型将拥有完整的特征金字塔交互能力，预计在 Recall 和 mAP50 上会有显著提升。