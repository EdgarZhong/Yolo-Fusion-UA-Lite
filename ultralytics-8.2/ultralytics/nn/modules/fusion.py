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
