import torch
import torch.nn as nn

from .conv import Conv


class Inception(nn.Module):
    """Inception 多分支模块：强制输入通道能被4整除，输出与输入通道一致"""

    def __init__(self, c1: int):
        super().__init__()
        # 强制检查：输入通道必须为4的倍数，否则抛出异常
        assert c1 % 4 == 0, f"FusionAttention.Inception 输入通道必须能被4整除，当前为 c1={c1}"
        c_branch = c1 // 4
        self.b1 = Conv(c1, c_branch, k=1, s=1)
        self.b2 = Conv(c1, c_branch, k=3, s=1, p=1)
        self.b3 = Conv(c1, c_branch, k=5, s=1, p=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4 = Conv(c1, c_branch, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.b1(x)
        b = self.b2(x)
        c = self.b3(x)
        d = self.b4(self.pool(x))
        return torch.cat([a, b, c, d], dim=1).contiguous()


class SEBlock(nn.Module):
    """通道注意力（全卷积实现），避免 AMP 下的 inplace 激活"""

    def __init__(self, c1: int, ratio: int = 16):
        super().__init__()
        c_mid = max(1, c1 // ratio)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c1, c_mid, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.SiLU(inplace=False)
        self.fc2 = nn.Conv2d(c_mid, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg(x)
        w = self.fc2(self.act(self.fc1(w)))
        w = self.gate(w)
        return x * w


class FusionAttention(nn.Module):
    """双模态融合（RGB/IR）：Inception+SE 后逐元素相加，输出通道与输入一致"""

    def __init__(self, c1: int):
        super().__init__()
        self.inc_rgb = Inception(c1)
        self.inc_ir = Inception(c1)
        self.se_rgb = SEBlock(c1)
        self.se_ir = SEBlock(c1)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_rgb, x_ir = x
        fr = self.se_rgb(self.inc_rgb(x_rgb))
        fi = self.se_ir(self.inc_ir(x_ir))
        return (fr + fi).contiguous()


class FeatureAttentionConcat(nn.Module):
    """改进版特征注意力拼接模块（FA-Concat）

    设计目标：
    - 仅做特征增强与去噪：分别对 RGB 与 IR 进行 Inception 多分支特征提取与 SE 通道注意力加权；
    - 不做融合（不逐元素相加）：避免信息的不可逆丢失；
    - 最终输出沿通道维进行拼接（concat），通道数为单模态的 2 倍；
    - 从输入/输出形态上与基线中的纯 Concat 保持一致，方便替换与对比。
    """

    def __init__(self, c1: int):
        """构造函数

        参数：
        - c1：单模态输入通道数（必须可被 4 整除，约束由 Inception 内部断言保证）
        """
        super().__init__()
        # 两路模态共享相同的增强结构：Inception + SE
        self.inc_rgb = Inception(c1)
        self.inc_ir = Inception(c1)
        self.se_rgb = SEBlock(c1)
        self.se_ir = SEBlock(c1)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """前向计算

        输入：
        - x：包含两路模态特征的列表或元组 (x_rgb, x_ir)，形状均为 [B, C, H, W]

        输出：
        - 沿通道维拼接后的张量，形状为 [B, 2*C, H, W]
        """
        x_rgb, x_ir = x
        # 对两路模态各自进行特征增强与噪声抑制
        fr = self.se_rgb(self.inc_rgb(x_rgb))  # 增强后的 RGB 特征
        fi = self.se_ir(self.inc_ir(x_ir))    # 增强后的 IR 特征
        # 不做加法融合，直接沿通道拼接，保持信息可逆与完整
        return torch.cat([fr, fi], dim=1).contiguous()
