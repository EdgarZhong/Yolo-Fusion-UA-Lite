# 工程组件升级编码实施手册：WaveletC2f + CrossModalAlign
> 研究端交付给编码 Agent
> 日期：2026-03-23
> **状态：模块实现可提前，集成到 YAML 需等注意力选型结果**

---

## 总览

本次实现两个独立模块。两者互不依赖，可并行开发。

| 模块 | 对应点子 | 作用 | 放置位置 |
|------|---------|------|---------|
| WaveletC2f | 点子 07 | IR 骨干频域先验增强 | 替换 IR 骨干中的 C2f 块 |
| CrossModalAlign | 点子 20 | P3 融合前的跨模态空间对齐 | P3 融合模块之前，新增一层 |

两个模块现在只实现代码和单元测试，**不集成到任何训练 YAML 中**——集成需要等注意力选型实验完成后，确定最终基线架构再做。

---

## 模块一：WaveletC2f

### 1.1 研究背景（简要）

IR（红外）图像的物理特性本质上是频域的——车辆与背景的热辐射差异体现在特定频段的对比度上。标准 CNN 的卷积核对这类信号的提取效率不如显式频域处理。WaveletC2f 在 C2f 块旁边并联一条 Haar 小波分支，将特征显式分解为低频（平滑轮廓）和高频（边缘纹理）子带，和 C2f 输出拼接后经投影层还原通道数。

小波滤波器权重固定（`requires_grad=False`），作为物理先验注入，不参与梯度更新。只有投影层是可学习的。

### 1.2 模块接口

```python
class WaveletC2f(nn.Module):
    """带 Haar 小波并联分支的 C2f 块，用于 IR 骨干"""
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        # c1: 输入通道数
        # c2: 输出通道数
        # n, shortcut, g, e: 与标准 C2f 参数含义一致
```

- 输入：`[B, c1, H, W]`
- 输出：`[B, c2, H, W]`
- 接口与标准 C2f **完全一致**，可在 YAML 中直接替换 `C2f`

### 1.3 内部结构

```
输入 [B, c1, H, W]
  ├─→ C2f(c1, c2, n, shortcut, g, e)  →  [B, c2, H, W]     ← 标准 C2f 分支
  │
  └─→ HaarWavelet2D(c1)               →  [B, 4*c1, H/2, W/2] ← 小波分支
       ↓ nn.Upsample(scale_factor=2)  →  [B, 4*c1, H, W]     ← 恢复空间分辨率
  
  concat([C2f输出, 小波输出], dim=1)   →  [B, c2 + 4*c1, H, W]
  ↓ Conv(c2 + 4*c1, c2, k=1, s=1)    →  [B, c2, H, W]       ← 投影层（可学习）
```

### 1.4 HaarWavelet2D 子模块

Haar 小波变换用固定权重的卷积实现，不需要外部库依赖。

**原理：** 2D Haar 小波将输入分解为 4 个子带：
- LL（低频-低频）：平滑近似，保留整体亮度和轮廓
- LH（低频-高频）：水平边缘
- HL（高频-低频）：垂直边缘  
- HH（高频-高频）：对角纹理/噪声

对于每个输入通道，用 4 组固定的 2×2 卷积核（stride=2）提取这 4 个子带。C 个输入通道产生 4C 个输出通道，空间尺寸减半。

**Haar 滤波器系数（固定，不可学习）：**

```python
# 每个滤波器是 2×2 的卷积核，系数如下：
LL = [[0.5,  0.5],  [0.5,  0.5]]   # 均值（低通-低通）
LH = [[0.5,  0.5],  [-0.5, -0.5]]  # 水平差分（低通-高通）
HL = [[0.5, -0.5],  [0.5, -0.5]]   # 垂直差分（高通-低通）
HH = [[0.5, -0.5],  [-0.5, 0.5]]   # 对角差分（高通-高通）
```

**实现方式：** 使用 `nn.Conv2d` 的分组卷积（`groups=c1`）实现逐通道变换。构造一个 `[4*c1, 1, 2, 2]` 的权重张量，每个输入通道对应 4 个滤波器。权重在 `__init__` 中设置并 `register_buffer`（不作为 parameter，不参与梯度，但跟随模型设备移动）。

```python
class HaarWavelet2D(nn.Module):
    """固定 Haar 小波 2D 分解，零可学习参数"""
    def __init__(self, c1: int):
        super().__init__()
        # 构造 Haar 滤波器权重 [4*c1, 1, 2, 2]
        # 对每个输入通道，4 个滤波器（LL, LH, HL, HH）
        filters = torch.tensor([[0.5, 0.5, 0.5, 0.5],
                                [0.5, 0.5, -0.5, -0.5],
                                [0.5, -0.5, 0.5, -0.5],
                                [0.5, -0.5, -0.5, 0.5]], dtype=torch.float32)
        # reshape 为 [4, 1, 2, 2]
        filters = filters.reshape(4, 1, 2, 2)
        # 扩展为 [4*c1, 1, 2, 2]，每个输入通道复制一组
        filters = filters.repeat(c1, 1, 1, 1)
        self.register_buffer('filters', filters)
        self.groups = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分组卷积，stride=2
        return nn.functional.conv2d(x, self.filters, stride=2, groups=self.groups)
```

### 1.5 关键实现约束

| 约束 | 说明 |
|------|------|
| Haar 滤波器 `requires_grad=False` | 使用 `register_buffer` 而非 `nn.Parameter`，确保不参与梯度更新 |
| 小波分支空间恢复 | Haar 变换后空间减半，必须上采样恢复到和 C2f 输出一致的空间尺寸再 concat |
| 上采样方式 | `nn.Upsample(scale_factor=2, mode='nearest')`，最简单且无可学习参数 |
| 投影层 | 1×1 Conv，将 concat 后的通道数还原为 c2。这是唯一的新增可学习参数 |
| COCO 权重兼容 | C2f 子模块可以直接加载 COCO 预训练权重（结构不变）；投影层随机初始化 |
| 数值精度 | 小波变换操作在 AMP 混合精度下是安全的（简单的固定系数卷积），不需要强制 `.float()` |

### 1.6 参数量估算

对于 P3 层（nano 实际通道数 c1=c2=64）：
- C2f 分支：与标准 C2f 完全相同，无额外参数
- Haar 小波分支：零参数（固定权重）
- 投影层：`(c2 + 4*c1) * c2 = (64 + 256) * 64 = 20,480` 参数 ≈ 0.02M

总新增参数极小，对 6M 总量的影响可忽略。

### 1.7 初始化策略

- C2f 子模块：正常从 COCO 预训练权重加载（结构完全一致）
- 投影层：随机初始化（PyTorch 默认的 kaiming 初始化即可）
- Haar 滤波器：`register_buffer` 在模型构造时固定，不参与权重加载

### 1.8 单元测试要求

```python
# 1. shape 一致性：输出 shape 与标准 C2f 完全一致
x = torch.randn(2, 64, 80, 64)
wavelet_c2f = WaveletC2f(64, 64, n=1, shortcut=True)
standard_c2f = C2f(64, 64, n=1, shortcut=True)
assert wavelet_c2f(x).shape == standard_c2f(x).shape

# 2. Haar 滤波器不可学习
for name, buf in wavelet_c2f.named_buffers():
    if 'filters' in name:
        assert not buf.requires_grad

# 3. 前向传播不报错（包括 AMP 半精度）
with torch.cuda.amp.autocast():
    out = wavelet_c2f(x.cuda())
    assert out.dtype == torch.float16  # AMP 下输出应为半精度
```

---

## 模块二：CrossModalAlign

### 2.1 研究背景（简要）

DroneVehicle 的 RGB 和 IR 图像存在像素级空间错位（标定误差、拍摄角度差异、平台运动）。预处理阶段已剔除了 329 对严重不一致样本，但剩余样本仍存在弱错位。在错位位置上做特征拼接或注意力计算，混合的是不同目标或目标边缘与背景的特征。

CrossModalAlign 在 P3 融合点之前，用可变形卷积在特征层面做隐式对齐——网络自己学习每个空间位置的偏移量（offset），将 RGB 特征 warp 到 IR 坐标系下。不需要显式图像配准。

### 2.2 模块接口

```python
class CrossModalAlign(nn.Module):
    """跨模态特征对齐：用可变形卷积将 RGB 特征对齐到 IR 坐标系"""
    def __init__(self, c1: int):
        # c1: 单模态通道数
```

- 输入：`list[Tensor, Tensor]`（`[x_rgb, x_ir]`，各 `[B, C, H, W]`）
- 输出：`list[Tensor, Tensor]`（`[x_rgb_aligned, x_ir]`，IR 不变，RGB 被对齐）

**⚠️ 注意：这个模块的输出格式是 `list[Tensor, Tensor]`，不是 `[B, 2C, H, W]`。** 它放在融合模块之前，输出的两路特征随后传入融合模块（FeatureAttentionConcat / InceptionCoordAttnConcat 等）做拼接。

### 2.3 内部结构

```
输入: [x_rgb, x_ir]  各 [B, C, H, W]

步骤 1: 计算 offset
  concat([x_rgb, x_ir], dim=1)      →  [B, 2C, H, W]
  ↓ Conv(2C, C, k=3, s=1, p=1)     →  [B, C, H, W]    ← offset 预测中间层
  ↓ Conv(C, 2*K*K, k=3, s=1, p=1)  →  [B, 2*K*K, H, W] ← offset 场（K=3）
  
步骤 2: 可变形卷积
  deform_conv2d(x_rgb, offset, weight) → [B, C, H, W]  ← 对齐后的 RGB 特征

输出: [x_rgb_aligned, x_ir]
```

其中 `K=3` 是可变形卷积的 kernel size，`2*K*K = 18` 表示每个空间位置的 9 个采样点各有 x/y 两个偏移量。

### 2.4 实现细节

**使用 `torchvision.ops.deform_conv2d`（原生支持，无需第三方库）：**

```python
import torchvision.ops

class CrossModalAlign(nn.Module):
    def __init__(self, c1: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # offset 预测网络：两路 concat 后预测空间偏移
        self.offset_conv1 = nn.Conv2d(2 * c1, c1, kernel_size=3, stride=1, padding=1)
        self.offset_bn = nn.BatchNorm2d(c1)
        self.offset_act = nn.SiLU(inplace=False)
        self.offset_conv2 = nn.Conv2d(c1, 2 * kernel_size * kernel_size, kernel_size=3, stride=1, padding=1)
        
        # 可变形卷积的权重（用于对 RGB 特征做空间变换）
        self.deform_weight = nn.Parameter(
            torch.randn(c1, c1, kernel_size, kernel_size) * 0.01
        )
        self.deform_padding = padding
        
        # offset 初始化为接近零（初始状态近似恒等变换）
        nn.init.zeros_(self.offset_conv2.weight)
        nn.init.zeros_(self.offset_conv2.bias)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        x_rgb, x_ir = x
        
        # 预测 offset
        combined = torch.cat([x_rgb, x_ir], dim=1)
        offset = self.offset_act(self.offset_bn(self.offset_conv1(combined)))
        offset = self.offset_conv2(offset)  # [B, 2*K*K, H, W]
        
        # 用可变形卷积对齐 RGB 特征
        x_rgb_aligned = torchvision.ops.deform_conv2d(
            input=x_rgb,
            offset=offset,
            weight=self.deform_weight,
            padding=self.deform_padding
        )
        
        return [x_rgb_aligned, x_ir]
```

### 2.5 关键实现约束

| 约束 | 说明 |
|------|------|
| **依赖** | `torchvision.ops.deform_conv2d`，torchvision ≥ 0.9.0（当前环境 0.21.0，已满足） |
| **offset 零初始化** | `offset_conv2` 的 weight 和 bias 都初始化为 0，初始状态下 offset 全零 → 可变形卷积退化为普通卷积 → 近似恒等变换。训练过程中网络逐渐学出非零 offset |
| **IR 不变** | 只对 RGB 做空间变换，IR 作为参照坐标系不动。选择 IR 作为参照是因为 IR 图像在本数据集中质量更稳定 |
| **kernel_size = 3** | 标准选择，每个输出位置参考 3×3=9 个可变形采样点，offset 通道数 = 2×9 = 18 |
| **SiLU `inplace=False`** | 遵循项目统一约束 |
| **输出格式** | `list[Tensor, Tensor]`，不是 concat 后的单张量——它是融合模块的前置步骤 |

### 2.6 在 YAML 中的使用方式（预览，暂不实施）

CrossModalAlign 作为融合模块的前置层插入。以 P3 融合为例：

```yaml
# 对齐层：输入 RGB-P3(7) 和 IR-P3(17)，输出对齐后的 [RGB-P3-aligned, IR-P3]
- [[7, 17], 1, CrossModalAlign, []]     # layer 23: 输出是 list[Tensor, Tensor]

# 融合层：输入对齐后的两路特征，输出 concat
- [-1, 1, InceptionCoordAttnConcat, []]  # layer 24: 输出 [B, 2C, H, W]

# C2f 规整
- [-1, 3, C2f, [256]]                   # layer 25: fused_P3
```

**⚠️ YAML 解析器兼容性：** `CrossModalAlign` 的输入是 `list[Tensor, Tensor]`（来自 YAML 多索引），输出也是 `list[Tensor, Tensor]`（传给下一个融合模块）。需要确认 ultralytics 的 YAML 解析器能否正确传递 list 类型的中间输出。如果不能，需要写一个包装类把两路打包/解包。agent 在实现时请验证这一点。

### 2.7 参数量估算

对于 P3 层（nano c1=64）：
- offset_conv1：`2*64 * 64 * 3 * 3 = 73,728` 参数
- offset_bn：`64 * 2 = 128` 参数
- offset_conv2：`64 * 18 * 3 * 3 = 10,368` 参数
- deform_weight：`64 * 64 * 3 * 3 = 36,864` 参数
- 总计：约 `121,088` 参数 ≈ 0.12M

对 6M 总量的影响约 2%，可接受。

### 2.8 初始化策略

- offset 网络：`offset_conv2` 零初始化（关键），其余层默认初始化
- `deform_weight`：小随机值（`* 0.01`），初始阶段近似恒等
- 整个模块不从 COCO 预训练加载——它是新增层，所有参数从随机/零开始

### 2.9 单元测试要求

```python
# 1. 输入输出 shape 一致（两路各自通道数和空间尺寸不变）
x_rgb = torch.randn(2, 64, 80, 64)
x_ir = torch.randn(2, 64, 80, 64)
align = CrossModalAlign(64)
out_rgb, out_ir = align([x_rgb, x_ir])
assert out_rgb.shape == x_rgb.shape
assert out_ir.shape == x_ir.shape

# 2. IR 不变（输出的 IR 和输入完全相同）
assert torch.equal(out_ir, x_ir)

# 3. 零初始化时近似恒等（offset 全零时输出应接近输入）
# 注：由于 deform_weight 有微小随机值，不会完全相等，但应该很接近
assert torch.allclose(out_rgb, x_rgb, atol=0.1)  # 宽松阈值

# 4. GPU + AMP 兼容
with torch.cuda.amp.autocast():
    out_rgb_amp, out_ir_amp = align.cuda()([x_rgb.cuda(), x_ir.cuda()])
    assert out_rgb_amp.shape == x_rgb.shape
```

---

## 模块注册

两个模块都需要：
1. 在 `fusion.py`（或新建 `wavelet.py`——由 agent 决定文件组织）中实现
2. 顶层模块在 `__init__.py` 中注册
3. 顶层模块在 `tasks.py` 解析逻辑中添加

`HaarWavelet2D` 是 `WaveletC2f` 的内部子模块，不需要单独注册到 YAML。

---

## 暂不实施的内容

以下内容等注意力选型完成后再做：

- WaveletC2f 集成到 IR 骨干的 YAML 配置
- CrossModalAlign 集成到 P3 融合前的 YAML 配置
- 包含这两个模块的训练脚本
- 消融实验设计

当前只实现模块代码 + 单元测试 + 模块注册，确保模块可用且正确。
