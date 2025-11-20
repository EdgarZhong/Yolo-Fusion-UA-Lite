import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k//2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ChannelGate(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        mid = max(c // r, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(self.pool(x))

class SpatialGate(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = ConvBNAct(c, c, k=3, g=c, act=True)
        self.proj = nn.Conv2d(c, 1, 1, bias=True)
    def forward(self, x):
        s = self.dw(x)
        return torch.sigmoid(self.proj(s))

class FusionAttentionBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.rgb_proj = ConvBNAct(c, c, k=1)
        self.ir_proj  = ConvBNAct(c, c, k=1)
        self.rgb_cg = ChannelGate(c, r)
        self.ir_cg  = ChannelGate(c, r)
        self.rgb_sg = SpatialGate(c)
        self.ir_sg  = SpatialGate(c)
        self.mix = ConvBNAct(c*2, c, k=1)

    def forward(self, x_rgb, x_ir):
        rgb = self.rgb_proj(x_rgb)
        ir  = self.ir_proj(x_ir)
        g_rgb, g_ir = self.rgb_cg(rgb), self.ir_cg(ir)
        s_rgb, s_ir = self.rgb_sg(rgb), self.ir_sg(ir)
        alpha = torch.sigmoid(g_rgb*(1+s_rgb) - g_ir*(1+s_ir))
        fused = alpha*rgb + (1-alpha)*ir
        return self.mix(torch.cat([fused, rgb+ir], dim=1))

class FusionAttention(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.blocks = nn.ModuleList([FusionAttentionBlock(c, r) for c in channels])
    def forward(self, xs_rgb, xs_ir):
        return [blk(a, b) for blk, a, b in zip(self.blocks, xs_rgb, xs_ir)]
