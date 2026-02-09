# 从 FA-Concat 迁移至 CM-FA-Concat 实施指导手册

**目标模块名称**：`CrossModalFusionAttention` (CM-FA-Concat)
**前置基础**：开发者已掌握 `FeatureAttentionConcat` (FA-Concat) 的实现逻辑。

---

### 1. 核心设计变更说明

从 **FA-Concat** 到 **CM-FA-Concat** 的核心转变在于**注意力机制的视野范围**。

*   **原 FA-Concat (独立增强)**：
    *   RGB 和 IR 模态各自拥有独立的 SE 模块。
    *   RGB 的权重仅由 RGB 特征决定，IR 的权重仅由 IR 特征决定。
    *   **缺陷**：无法处理“模态冲突”。例如，当 RGB 为全黑或高噪时，模型无法通过 IR 的清晰度来抑制 RGB 的权重。

*   **新 CM-FA-Concat (联合感知)**：
    *   引入 **Cross-Modal SE (跨模态 SE)** 单元。
    *   权重计算基于 RGB 和 IR 的**联合全局信息**。
    *   **优势**：实现“不确定性感知”。网络能同时“看到”两个模态的质量，从而动态分配权重（例如：在夜间场景自动抑制 RGB 通道权重，放大 IR 通道权重）。

---

### 2. 模块架构图

请参照下图理解数据流向的变更。注意 **SE 模块** 的变化。

#### 2.1 顶层架构对比

**旧版 (FA-Concat):**
```text
输入 [RGB, IR]
    |      |
    v      v
[Inception] [Inception]  <-- 独立特征提取
    |          |
    v          v
 [SE_RGB]   [SE_IR]      <-- 独立计算权重 (互不干扰)
    |          |
    v          v
  加权RGB    加权IR
    \        /
     \      /
      Concat             <-- 拼接输出 (通道数 2C)
```

**新版 (CM-FA-Concat):**
```text
输入 [RGB, IR]
    |      |
    v      v
[Inception] [Inception]  <-- 独立特征提取 (保持不变)
    |          |
    \          /
     \        /
   [CrossModalSE]        <-- 核心变更：联合计算权重 (交互发生处)
     /        \
    /          \
  加权RGB    加权IR
    \        /
     \      /
      Concat             <-- 拼接输出 (通道数 2C，接口保持一致)
```

---

### 3. 核心子模块：CrossModalSE 内部逻辑

这是你需要新编写或重构的子模块。它不再是单入单出，而是**双入双出**。

#### 3.1 逻辑流程
1.  **输入接收**：同时接收 RGB 特征图 (`Feat_RGB`) 和 IR 特征图 (`Feat_IR`)。
2.  **全局压缩 (Squeeze)**：分别对两个特征图进行全局平均池化，获得两个特征向量。
3.  **早期融合 (Early Fusion)**：**关键步骤**。将两个特征向量在通道维度拼接，形成一个包含双模态信息的“全局上下文向量”。
4.  **联合推理 (Excitation)**：
    *   将“全局上下文向量”送入全连接层（或 1x1 卷积）。
    *   网络此时拥有“上帝视角”，能根据两者的对比情况学习权重的分配策略。
5.  **权重分发**：输出一个长度为 `2 * C` 的权重向量，并将其拆分为 `Weight_RGB` 和 `Weight_IR`。
6.  **校准输出**：
    *   `Output_RGB = Feat_RGB * Weight_RGB`
    *   `Output_IR = Feat_IR * Weight_IR`

#### 3.2 结构图示
```text
[Feat_RGB]      [Feat_IR]      (形状: B, C, H, W)
    |               |
  Pool            Pool         (全局平均池化)
    |               |
 [Vec_R]         [Vec_I]       (形状: B, C, 1, 1)
    \             /
     \           /
      \         /
       Concat                  (形状: B, 2C, 1, 1) <-- 跨模态信息在此汇聚
          |
     FC Layers (MLP)           (全连接/1x1卷积，含降维与升维)
          |
       Sigmoid                 (激活)
          |
     [Joint_Weights]           (形状: B, 2C, 1, 1)
          |
        Split                  (拆分)
       /     \
 [Weight_R] [Weight_I]         (形状: B, C, 1, 1)
    |          |
    x          x               (乘法广播)
 [Feat_RGB]  [Feat_IR]
```

---

### 4. 实施步骤指南

请按照以下逻辑修改代码结构：

1.  **复用 Inception**：保持原有的 `Inception` 类完全不变。
2.  **重构 SE 模块**：
    *   实现新的 `CrossModalSE` 类。
    *   **参数变化**：`__init__` 中定义的中间层（降维层）通道数应基于 `2 * C` 计算，而不是 `C`。
3.  **重写主类 (CrossModalFusionAttention)**：
    *   **初始化 (`__init__`)**：
        *   实例化两个 `Inception` 对象（分别对应 RGB/IR）。
        *   实例化**一个** `CrossModalSE` 对象（共享使用）。
    *   **前向传播 (`forward`)**：
        *   步骤 1：分别通过 `Inception` 提取特征。
        *   步骤 2：将两个特征**同时**传入 `CrossModalSE`，直接获得两个加权后的特征。
        *   步骤 3：执行 `torch.cat` 进行拼接。

### 5. 接口一致性检查

为确保能无缝替换现有的 FA-Concat 模块，请确保满足以下接口约束：

*   **输入**：必须接受包含两个张量的列表/元组 `[x_rgb, x_ir]`。
*   **输出**：必须是一个张量，且通道数为输入单模态通道数的 **2倍** (`2 * C1`)。
*   **后续层**：由于输出依然是拼接形式 (`Concat`)，后续连接的 `C2f` 模块无需修改输入通道配置（YOLO 框架会自动计算上一层的输出通道数）。