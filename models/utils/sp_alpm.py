import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencd.registry import MODELS


import torch
import torch.nn as nn
import torch.nn.functional as F

# class FFTDecomposer(nn.Module):
#     """稳定替代版：空间域高低频分解，不使用 torch.fft"""
#
#     def __init__(self, channels=3, kernel_size=5, sigma=1.0):
#         super().__init__()
#         self.channels = channels
#         self.kernel_size = kernel_size
#         self.sigma = sigma
#
#         kernel = self._build_gaussian_kernel(kernel_size, sigma)
#         kernel = kernel.repeat(channels, 1, 1, 1)
#         self.register_buffer("kernel", kernel)
#
#     def _build_gaussian_kernel(self, k, sigma):
#         ax = torch.arange(k).float() - (k - 1) / 2
#         xx, yy = torch.meshgrid(ax, ax, indexing='ij')
#         kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
#         kernel = kernel / kernel.sum()
#         return kernel[None, None, :, :]
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = x.float().contiguous()
#
#         if C != self.channels:
#             raise ValueError(f"Expected {self.channels} channels, but got {C}")
#
#         low = F.conv2d(
#             x,
#             self.kernel.to(device=x.device, dtype=x.dtype),
#             padding=self.kernel_size // 2,
#             groups=C
#         )
#         high = x - low
#         return low, high

class FFTDecomposer(nn.Module):
    """对浅层特征做 FFT 高低频分解。"""

    def __init__(self, low_freq_radius=0.5):
        super().__init__()
        self.low_freq_radius = low_freq_radius

    def _build_radial_mask(self, h, w, device):
        y = torch.linspace(-1.0, 1.0, steps=h, device=device).view(h, 1)
        x = torch.linspace(-1.0, 1.0, steps=w, device=device).view(1, w)
        dist = torch.sqrt(x ** 2 + y ** 2)
        mask = (dist <= self.low_freq_radius).float()
        return mask[None, None, :, :]  # [1,1,H,W]

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        #fft = torch.fft.fft2(x, norm='ortho')
        
        # fft = torch.fft.fft2(x.detach().cpu(), norm='ortho').to(x.device)
        # fft = torch.fft.fftshift(fft, dim=(-2, -1))

        x_cpu = x.detach().to('cpu')
        fft = torch.fft.fft2(x_cpu, norm='ortho')

        H, W = fft.shape[-2:]
        fft = torch.roll(fft, shifts=(H // 2, W // 2), dims=(-2, -1))

        fft = fft.to(x.device)

        low_mask = self._build_radial_mask(H, W, x.device)
        high_mask = 1.0 - low_mask

        low_fft = fft * low_mask
        high_fft = fft * high_mask

        low_fft = torch.fft.ifftshift(low_fft, dim=(-2, -1))
        high_fft = torch.fft.ifftshift(high_fft, dim=(-2, -1))

        # low = torch.fft.ifft2(low_fft, norm='ortho').real
        # high = torch.fft.ifft2(high_fft, norm='ortho').real

        low_fft_cpu = low_fft.detach().to('cpu')
        high_fft_cpu = high_fft.detach().to('cpu')

        H, W = low_fft_cpu.shape[-2:]

        low_fft_cpu = torch.roll(low_fft_cpu, shifts=(-H // 2, -W // 2), dims=(-2, -1))
        high_fft_cpu = torch.roll(high_fft_cpu, shifts=(-H // 2, -W // 2), dims=(-2, -1))

        low = torch.fft.ifft2(low_fft_cpu, norm='ortho').real.to(low_fft.device)
        high = torch.fft.ifft2(high_fft_cpu, norm='ortho').real.to(high_fft.device)
        
        return low, high

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


class ChangeMask(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, d):
        return self.conv(d)

class SPALPM(nn.Module):
    """
    输入双时相高频特征 h1, h2
    输出增强后的双时相高频特征和结构保持损失
    """
    def __init__(self, channels, levels=2):
        super().__init__()
        self.levels = levels
        self.pyramid = GaussianPyramid()

        self.scale_gates = nn.ModuleList([
            ScaleGate(channels) for _ in range(levels)
        ])
        self.change_masks = nn.ModuleList([
            ChangeMask(channels) for _ in range(levels)
        ])

    def reconstruct(self, g_last_1, g_last_2, p1_tilde, p2_tilde):
        cur1, cur2 = g_last_1, g_last_2
        for l in reversed(range(self.levels)):
            cur1 = p1_tilde[l] + F.interpolate(
                cur1, size=p1_tilde[l].shape[-2:], mode='bilinear', align_corners=False)
            cur2 = p2_tilde[l] + F.interpolate(
                cur2, size=p2_tilde[l].shape[-2:], mode='bilinear', align_corners=False)
        return cur1, cur2

    def forward(self, h1, h2):
        g1, p1 = self.pyramid.build_laplacian_pyramid(h1, self.levels)
        g2, p2 = self.pyramid.build_laplacian_pyramid(h2, self.levels)

        p1_tilde, p2_tilde = [], []
        loss_sp = 0.0

        for l in range(self.levels):
            a1 = self.scale_gates[l](p1[l])
            a2 = self.scale_gates[l](p2[l])

            p1_hat = a1 * p1[l]
            p2_hat = a2 * p2[l]

            d = torch.abs(p1[l] - p2[l])
            m = self.change_masks[l](d)

            p1_enh = m * p1_hat
            p2_enh = m * p2_hat

            p1_tilde.append(p1_enh)
            p2_tilde.append(p2_enh)

            # 结构保持正则
            loss_sp = loss_sp + F.l1_loss(p1_enh, p1[l]) + F.l1_loss(p2_enh, p2[l])

        out1, out2 = self.reconstruct(g1[-1], g2[-1], p1_tilde, p2_tilde)
        return out1, out2, loss_sp

class FreqFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, low, high):
        return self.fuse(torch.cat([low, high], dim=1))

@MODELS.register_module()
class SPALPMFrontEnd(nn.Module):
    """
    输入: img1, img2  [B,3,H,W]
    输出: feat1, feat2, loss_sp
    """
    def __init__(self, in_channels=3, feat_channels=3, levels=2, low_freq_radius=0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        #self.fft = FFTDecomposer()
        self.fft = FFTDecomposer(low_freq_radius=low_freq_radius)
        self.spalpm = SPALPM(channels=feat_channels, levels=levels)
        self.fusion = FreqFusion(channels=feat_channels)

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, img1, img2):
        f1 = self.stem(img1)
        f2 = self.stem(img2)

        low1, high1 = self.fft(f1)
        low2, high2 = self.fft(f2)

        high1_enh, high2_enh, loss_sp = self.spalpm(high1, high2)

        out1 = f1 + self.alpha * self.fusion(low1, high1_enh)
        out2 = f2 + self.alpha * self.fusion(low2, high2_enh)
        return out1, out2, loss_sp