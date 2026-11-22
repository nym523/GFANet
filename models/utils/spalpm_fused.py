import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [[1., 4., 6., 4., 1.],
             [4.,16.,24.,16., 4.],
             [6.,24.,36.,24., 6.],
             [4.,16.,24.,16., 4.],
             [1., 4., 6., 4., 1.]]
        )
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None, :, :])

    def gaussian_blur(self, x):
        c = x.shape[1]
        kernel = self.kernel.repeat(c, 1, 1, 1)
        return F.conv2d(x, kernel, padding=2, groups=c)

    def down(self, x):
        return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

    def up(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def build_gaussian_pyramid(self, x, levels):
        gs = [x]
        cur = x
        for _ in range(levels):
            cur = self.down(self.gaussian_blur(cur))
            gs.append(cur)
        return gs

    def build_laplacian_pyramid(self, x, levels):
        gs = self.build_gaussian_pyramid(x, levels)
        ps = []
        for l in range(levels):
            up = self.up(gs[l + 1], gs[l].shape[-2:])
            ps.append(gs[l] - up)
        return gs, ps


class ScaleGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)


class SPALPMFused(nn.Module):
    """
    单输入版：用于 decode head 中 fusion_conv 之后
    输入: x [B, C, H, W]
    输出: x_enh, loss_sp
    """
    def __init__(self, channels, levels=1, beta=0.1):
        super().__init__()
        self.levels = levels
        self.beta = beta
        self.pyramid = GaussianPyramid()
        self.scale_gates = nn.ModuleList([
            ScaleGate(channels) for _ in range(levels)
        ])

    def reconstruct(self, g_last, p_tilde):
        cur = g_last
        for l in reversed(range(self.levels)):
            cur = p_tilde[l] + F.interpolate(
                cur, size=p_tilde[l].shape[-2:], mode='bilinear', align_corners=False)
        return cur

    def forward(self, x):
        g, p = self.pyramid.build_laplacian_pyramid(x, self.levels)

        p_tilde = []
        loss_sp = 0.0

        for l in range(self.levels):
            a = self.scale_gates[l](p[l])
            p_enh = p[l] + self.beta * (a * p[l])   # 残差式增强
            p_tilde.append(p_enh)
            loss_sp = loss_sp + F.l1_loss(p_enh, p[l])

        out = self.reconstruct(g[-1], p_tilde)
        return out, loss_sp