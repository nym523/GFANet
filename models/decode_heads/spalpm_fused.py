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


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class FrequencyPrompt(nn.Module):
    def __init__(self, channels, reduction=4, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.amp_gate = ChannelGate(channels, reduction)
        self.phase_gate = ChannelGate(channels, reduction)

    def forward(self, x):
        x = x.contiguous()
        orig_dtype = x.dtype
        x_fft = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

        try:
            fft = torch.fft.fft2(x_fft, norm='ortho')
        except RuntimeError as e:
            if 'cuFFT' in str(e) or 'CUFFT' in str(e):
                x_cpu = x_fft.detach().cpu()
                fft = torch.fft.fft2(x_cpu, norm='ortho').to(x.device)
            else:
                raise e

        amp = torch.log1p(torch.abs(fft))
        phase = torch.atan2(fft.imag, fft.real + self.eps)

        amp = amp.to(orig_dtype)
        phase = phase.to(orig_dtype)

        m_amp = self.amp_gate(amp)
        m_phase = self.phase_gate(phase)
        return m_amp, m_phase

class SPALPMFused(nn.Module):
    """
    升级版：
    - 单输入：融合后的差异特征
    - 拉普拉斯金字塔分解
    - 幅度-相位解耦门控
    - 残差式结构保持重建
    """
    def __init__(
        self,
        channels,
        levels=1,
        beta=0.1,
        gamma=0.1,
        reduction=4
    ):
        super().__init__()
        self.levels = levels
        self.beta = beta
        self.gamma = gamma

        self.pyramid = GaussianPyramid()
        self.freq_prompt = FrequencyPrompt(channels, reduction=reduction)

        self.amp_align = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            for _ in range(levels)
        ])
        self.phase_align = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            for _ in range(levels)
        ])

        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def reconstruct(self, g_last, p_tilde):
        cur = g_last
        for l in reversed(range(self.levels)):
            cur = p_tilde[l] + F.interpolate(
                cur, size=p_tilde[l].shape[-2:], mode='bilinear', align_corners=False
            )
        return cur

    def forward(self, x):
        g, p = self.pyramid.build_laplacian_pyramid(x, self.levels)

        m_amp, m_phase = self.freq_prompt(x)   # [B, C, 1, 1]

        p_tilde = []
        loss_sp = 0.0

        for l in range(self.levels):
            m_amp_l = torch.sigmoid(self.amp_align[l](m_amp))
            m_phase_l = torch.sigmoid(self.phase_align[l](m_phase))

            p_enh = (
                p[l]
                + self.beta * (m_phase_l * p[l])
                - self.gamma * (m_amp_l * p[l])
            )

            p_tilde.append(p_enh)
            loss_sp = loss_sp + F.l1_loss(p_enh, p[l])

        out_rec = self.reconstruct(g[-1], p_tilde)

        # 残差式输出，增强训练稳定性
        out = x + self.out_proj(out_rec - x)

        return out, loss_sp