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


class InceptionConcat(nn.Module):
    def __init__(self, c1: int):
        super().__init__()
        self.inc_rgb = Inception(c1)
        self.inc_ir = Inception(c1)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_rgb, x_ir = x
        fr = self.inc_rgb(x_rgb)
        fi = self.inc_ir(x_ir)
        return torch.cat([fr, fi], dim=1).contiguous()


class h_sigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp: int, oup: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return (identity * a_h * a_w).contiguous()


class SimAM(nn.Module):
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return (x * torch.sigmoid(e_inv)).contiguous()


class InceptionCoordAttnConcat(nn.Module):
    def __init__(self, c1: int):
        super().__init__()
        self.inc_rgb = Inception(c1)
        self.inc_ir = Inception(c1)
        self.ca_rgb = CoordAtt(c1, c1)
        self.ca_ir = CoordAtt(c1, c1)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_rgb, x_ir = x
        fr = self.ca_rgb(self.inc_rgb(x_rgb))
        fi = self.ca_ir(self.inc_ir(x_ir))
        return torch.cat([fr, fi], dim=1).contiguous()


class InceptionSimAMConcat(nn.Module):
    def __init__(self, c1: int):
        super().__init__()
        self.inc_rgb = Inception(c1)
        self.inc_ir = Inception(c1)
        self.sa_rgb = SimAM()
        self.sa_ir = SimAM()

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_rgb, x_ir = x
        fr = self.sa_rgb(self.inc_rgb(x_rgb))
        fi = self.sa_ir(self.inc_ir(x_ir))
        return torch.cat([fr, fi], dim=1).contiguous()


class CrossModalSE(nn.Module):
    """跨模态 SE 注意力模块：联合感知 RGB 与 IR 的全局信息后分别生成两路权重

    设计动机：
    - 传统 SE 为单入单出，权重仅由本模态特征决定，无法在 RGB/IR 质量不一致时做抑制/放大；
    - 跨模态 SE 同时接收两路特征，先做全局池化得到两个描述符，再在通道维进行早期拼接，
      通过 1x1 卷积 MLP 联合推理，输出长度为 2*C 的权重，并拆分回两路分别加权。
    """

    def __init__(self, c1: int, ratio: int = 16):
        super().__init__()
        # 这里的中间层通道数基于联合通道数 2*C 进行压缩，避免过拟合同时保留足够表达能力
        c_joint = 2 * c1
        c_mid = max(1, c_joint // ratio)
        # 全局平均池化将每路特征压缩为 [B, C, 1, 1]
        self.avg = nn.AdaptiveAvgPool2d(1)
        # 1x1 全卷积实现的两层 MLP：降维 -> SiLU -> 升维 -> Sigmoid 门控
        self.fc1 = nn.Conv2d(c_joint, c_mid, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.SiLU(inplace=False)
        self.fc2 = nn.Conv2d(c_mid, c_joint, kernel_size=1, stride=1, padding=0, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x_rgb: torch.Tensor, x_ir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向：联合两路全局信息，生成并分发权重到各自模态

        输入：
        - x_rgb, x_ir: 两路特征，形状均为 [B, C, H, W]

        输出：
        - 分别加权后的两路特征，形状与输入一致
        """
        # 分别做全局平均池化得到两路描述符
        r = self.avg(x_rgb)
        i = self.avg(x_ir)
        # 在通道维度早期融合，形成联合上下文向量 [B, 2*C, 1, 1]
        joint = torch.cat([r, i], dim=1).contiguous()
        # 联合推理得到长度为 2*C 的权重向量，并用 Sigmoid 映射到 0~1
        w = self.fc2(self.act(self.fc1(joint)))
        w = self.gate(w)
        # 拆分并分发到两路模态，按通道广播相乘
        w_r, w_i = torch.split(w, x_rgb.size(1), dim=1)
        return (x_rgb * w_r).contiguous(), (x_ir * w_i).contiguous()


class CrossModalFusionAttention(nn.Module):
    """CM-FA-Concat：Inception 提取 + 跨模态 SE 加权 + 通道拼接

    接口保持与 FA‑Concat 一致：输入为 [RGB, IR]，输出为通道维拼接后的张量，通道数为 2*C。
    差异在于权重计算由 CrossModalSE 统一建模两路质量，从而在夜间/雾天等场景自动抑制低质量模态。
    """

    def __init__(self, c1: int):
        super().__init__()
        # 两路独立 Inception 特征提取（保持不变）
        self.inc_rgb = Inception(c1)
        self.inc_ir = Inception(c1)
        # 单个跨模态 SE 单元共享使用
        self.cm_se = CrossModalSE(c1)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """前向：先独立提取特征，再做跨模态权重分配，最终拼接输出"""
        x_rgb, x_ir = x
        fr = self.inc_rgb(x_rgb)
        fi = self.inc_ir(x_ir)
        fr_w, fi_w = self.cm_se(fr, fi)
        return torch.cat([fr_w, fi_w], dim=1).contiguous()
