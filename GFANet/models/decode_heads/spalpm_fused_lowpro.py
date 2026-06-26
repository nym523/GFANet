# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from pytorch_wavelets import DTCWTForward
#
#
# # =========================================================
# # 1. High-Frequency Enhancement Branch
# # =========================================================
# class GaussianPyramid(nn.Module):
#     def __init__(self):
#         super().__init__()
#         kernel = torch.tensor(
#             [[1., 4., 6., 4., 1.],
#              [4., 16., 24., 16., 4.],
#              [6., 24., 36., 24., 6.],
#              [4., 16., 24., 16., 4.],
#              [1., 4., 6., 4., 1.]]
#         )
#         kernel = kernel / kernel.sum()
#         self.register_buffer("kernel", kernel[None, None, :, :])
#
#     def gaussian_blur(self, x):
#         c = x.shape[1]
#         kernel = self.kernel.repeat(c, 1, 1, 1)
#         return F.conv2d(x, kernel, padding=2, groups=c)
#
#     def down(self, x):
#         return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
#
#     def up(self, x, size):
#         return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
#
#     def build_gaussian_pyramid(self, x, levels):
#         gs = [x]
#         cur = x
#         for _ in range(levels):
#             cur = self.down(self.gaussian_blur(cur))
#             gs.append(cur)
#         return gs
#
#     def build_laplacian_pyramid(self, x, levels):
#         gs = self.build_gaussian_pyramid(x, levels)
#         ps = []
#         for l in range(levels):
#             up = self.up(gs[l + 1], gs[l].shape[-2:])
#             ps.append(gs[l] - up)
#         return gs, ps
#
#
# class ChannelGate(nn.Module):
#     def __init__(self, channels, reduction=4):
#         super().__init__()
#         hidden = max(channels // reduction, 4)
#         self.net = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, hidden, 1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden, channels, 1, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class FrequencyPrompt(nn.Module):
#     def __init__(self, channels, reduction=4, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#         self.amp_gate = ChannelGate(channels, reduction)
#         self.phase_gate = ChannelGate(channels, reduction)
#
#     def forward(self, x):
#         x = x.contiguous()
#         orig_dtype = x.dtype
#         x_fft = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
#
#         try:
#             fft = torch.fft.fft2(x_fft, norm='ortho')
#         except RuntimeError as e:
#             if 'cuFFT' in str(e) or 'CUFFT' in str(e):
#                 x_cpu = x_fft.detach().cpu()
#                 fft = torch.fft.fft2(x_cpu, norm='ortho').to(x.device)
#             else:
#                 raise e
#
#         amp = torch.log1p(torch.abs(fft))
#         phase = torch.atan2(fft.imag, fft.real + self.eps)
#
#         amp = amp.to(orig_dtype)
#         phase = phase.to(orig_dtype)
#
#         m_amp = self.amp_gate(amp)
#         m_phase = self.phase_gate(phase)
#         return m_amp, m_phase
#
#
# class SPALPMHighEnhancer(nn.Module):
#     """
#     高频增强支路：
#     - 拉普拉斯金字塔分解
#     - 幅度/相位门控
#     - 残差式结构保持增强
#     """
#     def __init__(
#         self,
#         channels,
#         levels=1,
#         beta=0.1,
#         gamma=0.1,
#         reduction=4,
#         alpha=0.1
#     ):
#         super().__init__()
#         self.levels = levels
#         self.beta = beta
#         self.gamma = gamma
#
#         self.pyramid = GaussianPyramid()
#         self.freq_prompt = FrequencyPrompt(channels, reduction=reduction)
#
#         self.amp_align = nn.ModuleList([
#             nn.Conv2d(channels, channels, kernel_size=1, bias=True)
#             for _ in range(levels)
#         ])
#         self.phase_align = nn.ModuleList([
#             nn.Conv2d(channels, channels, kernel_size=1, bias=True)
#             for _ in range(levels)
#         ])
#
#         self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
#
#         alpha = float(alpha)
#         alpha = min(max(alpha, 1e-4), 1 - 1e-4)
#         alpha_logit = math.log(alpha / (1 - alpha))
#         self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))
#
#     @property
#     def alpha(self):
#         return torch.sigmoid(self.alpha_logit)
#
#     def reconstruct(self, g_last, p_tilde):
#         cur = g_last
#         for l in reversed(range(self.levels)):
#             cur = p_tilde[l] + F.interpolate(
#                 cur, size=p_tilde[l].shape[-2:], mode='bilinear', align_corners=False
#             )
#         return cur
#
#     def forward(self, x):
#         g, p = self.pyramid.build_laplacian_pyramid(x, self.levels)
#
#         m_amp, m_phase = self.freq_prompt(x)
#
#         p_tilde = []
#         loss_sp = x.new_zeros(())
#
#         for l in range(self.levels):
#             m_amp_l = torch.sigmoid(self.amp_align[l](m_amp))
#             m_phase_l = torch.sigmoid(self.phase_align[l](m_phase))
#
#             p_enh = (
#                 p[l]
#                 + self.beta * (m_phase_l * p[l])
#                 - self.gamma * (m_amp_l * p[l])
#             )
#
#             p_tilde.append(p_enh)
#             loss_sp = loss_sp + F.l1_loss(p_enh, p[l])
#
#         out_rec = self.reconstruct(g[-1], p_tilde)
#
#         high_feat = x + self.alpha * self.out_proj(out_rec - x)
#
#         return high_feat, loss_sp
#
#
# # =========================================================
# # 2. Low-Frequency Calibration Branch
# #    DTCWT low-frequency extraction + ECA attention + adjacency CRF
# # =========================================================
# class DTCWTLowFrequencyExtractor(nn.Module):
#     def __init__(self, J=1, biort='near_sym_a', qshift='qshift_a'):
#         super().__init__()
#         self.xfm = DTCWTForward(J=J, biort=biort, qshift=qshift)
#
#     def forward(self, x):
#         yl, yh = self.xfm(x)
#         return yl
#
#
# class ECALayer(nn.Module):
#     """
#     ECA-Net channel attention
#     """
#     def __init__(self, channels, k_size=3):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(
#             1, 1,
#             kernel_size=k_size,
#             padding=(k_size - 1) // 2,
#             bias=False
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x: [B, C, H, W]
#         y = self.avg_pool(x)                              # [B, C, 1, 1]
#         y = y.squeeze(-1).transpose(-1, -2)              # [B, 1, C]
#         y = self.conv(y)                                 # [B, 1, C]
#         y = y.transpose(-1, -2).unsqueeze(-1)            # [B, C, 1, 1]
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)
#
#
# class ECAResidualBlock(nn.Module):
#     """
#     局部卷积 + ECA 通道重标定 + 残差连接
#     用于替代原来的 FFCResBlock
#     """
#     def __init__(self, channels, k_size=9):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channels)
#         )
#         self.eca = ECALayer(channels, k_size=k_size)
#         self.act = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         identity = x
#         out = self.conv(x)
#         out = self.eca(out)
#         out = out + identity
#         out = self.act(out)
#         return out
#
#
# class AdjCRFPosteriorRefiner(nn.Module):
#     """
#     在池化后的低频特征图上构建 8 邻域 CRF，
#     使用 mean-field 风格近似推断后验概率，
#     输出变化类概率图 prob_change。
#     """
#     def __init__(
#         self,
#         channels,
#         hidden_ratio=0.25,
#         num_iters=5,
#         sigma_feat=0.5,
#         sigma_pos=0.3,
#         max_hw=24
#     ):
#         super().__init__()
#         hidden = max(int(channels * hidden_ratio), 8)
#
#         self.num_iters = num_iters
#         self.sigma_feat = sigma_feat
#         self.sigma_pos = sigma_pos
#         self.max_hw = max_hw
#
#         self.embed_proj = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
#         self.unary_proj = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
#         self.classifier = nn.Conv2d(hidden, 2, kernel_size=1, bias=True)
#
#     @staticmethod
#     def _build_coords(h, w, device, dtype):
#         ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
#         xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
#         yy, xx = torch.meshgrid(ys, xs, indexing='ij')
#         coords = torch.stack([yy, xx], dim=0).unsqueeze(0)
#         return coords
#
#     @staticmethod
#     def _shift(x, dy, dx):
#         b, c, h, w = x.shape
#         pad_l = max(dx, 0)
#         pad_r = max(-dx, 0)
#         pad_t = max(dy, 0)
#         pad_b = max(-dy, 0)
#         x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0.0)
#
#         y1 = pad_b
#         y2 = y1 + h
#         x1 = pad_r
#         x2 = x1 + w
#         return x_pad[:, :, y1:y2, x1:x2]
#
#     @staticmethod
#     def _shift_mask(h, w, dy, dx, device, dtype):
#         mask = torch.ones(1, 1, h, w, device=device, dtype=dtype)
#         return AdjCRFPosteriorRefiner._shift(mask, dy, dx)
#
#     def _pairwise_kernel(self, feat, coords):
#         offsets = [(-1, -1), (-1, 0), (-1, 1),
#                    (0, -1),           (0, 1),
#                    (1, -1),  (1, 0),  (1, 1)]
#
#         weights = []
#         for dy, dx in offsets:
#             feat_n = self._shift(feat, dy, dx)
#             coord_n = self._shift(coords, dy, dx)
#
#             d_feat = ((feat - feat_n) ** 2).sum(dim=1, keepdim=True)
#             d_pos = ((coords - coord_n) ** 2).sum(dim=1, keepdim=True)
#
#             w = torch.exp(-d_feat - d_pos)
#
#             mask = self._shift_mask(
#                 feat.shape[-2], feat.shape[-1], dy, dx, feat.device, feat.dtype
#             )
#             w = w * mask
#             weights.append((dy, dx, w))
#
#         return weights
#
#     def _mean_field_inference(self, unary_logits, pairwise_weights):
#         q = torch.softmax(unary_logits, dim=1)
#
#         for _ in range(self.num_iters):
#             msg = 0.0
#             for dy, dx, w in pairwise_weights:
#                 q_n = self._shift(q, dy, dx)
#                 msg = msg + w * q_n
#             q = torch.softmax(unary_logits + msg, dim=1)
#
#         return q
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         hs = min(h, self.max_hw)
#         ws = min(w, self.max_hw)
#
#         if hs != h or ws != w:
#             x_small = F.adaptive_avg_pool2d(x, (hs, ws))
#         else:
#             x_small = x
#
#         feat = self.embed_proj(x_small)
#         unary_feat = self.unary_proj(x_small)
#         unary_logits = self.classifier(unary_feat)
#
#         coords = self._build_coords(hs, ws, x_small.device, x_small.dtype).expand(b, -1, -1, -1)
#         pairwise_weights = self._pairwise_kernel(feat, coords)
#
#         post_q = self._mean_field_inference(unary_logits, pairwise_weights)
#         prob_change_small = post_q[:, 1:2, :, :]
#
#         init_prob = torch.softmax(unary_logits, dim=1)[:, 1:2, :, :]
#         loss_lf = F.l1_loss(prob_change_small, init_prob.detach())
#
#         if hs != h or ws != w:
#             prob_change = F.interpolate(
#                 prob_change_small, size=(h, w), mode='bilinear', align_corners=False
#             )
#         else:
#             prob_change = prob_change_small
#
#         return prob_change, loss_lf
#
#
# class DTCWTLowRefiner(nn.Module):
#     """
#     低频校正支路：
#         Ffus -> DTCWT低频 -> ECA attention -> 邻接CRF
#              -> gamma map -> 通道映射 -> 低频校正结果 Flow
#     """
#     def __init__(
#         self,
#         channels,
#         dtcwt_J=1,
#         biort='near_sym_a',
#         qshift='qshift_a',
#         eca_k_size=3,
#         num_eca_blocks=1,
#         lf_alpha=0.1,
#         crf_hidden_ratio=0.25,
#         crf_num_iters=5,
#         crf_sigma_feat=0.5,
#         crf_sigma_pos=0.3,
#         crf_max_hw=24
#     ):
#         super().__init__()
#
#         self.low_extractor = DTCWTLowFrequencyExtractor(
#             J=dtcwt_J,
#             biort=biort,
#             qshift=qshift
#         )
#
#         self.low_proj = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True)
#         )
#
#         self.eca_blocks = nn.Sequential(*[
#             ECAResidualBlock(
#                 channels=channels,
#                 k_size=eca_k_size
#             ) for _ in range(num_eca_blocks)
#         ])
#
#         self.posterior_refiner = AdjCRFPosteriorRefiner(
#             channels=channels,
#             hidden_ratio=crf_hidden_ratio,
#             num_iters=crf_num_iters,
#             sigma_feat=crf_sigma_feat,
#             sigma_pos=crf_sigma_pos,
#             max_hw=crf_max_hw
#         )
#
#         self.gamma_proj = nn.Sequential(
#             nn.Conv2d(1, channels, kernel_size=1, bias=True),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True)
#         )
#
#         self.out_proj = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True)
#         )
#
#         lf_alpha = float(lf_alpha)
#         lf_alpha = min(max(lf_alpha, 1e-4), 1 - 1e-4)
#         lf_alpha_logit = math.log(lf_alpha / (1 - lf_alpha))
#         self.lf_alpha_logit = nn.Parameter(torch.tensor(lf_alpha_logit, dtype=torch.float32))
#
#     @property
#     def lf_alpha(self):
#         return torch.sigmoid(self.lf_alpha_logit)
#
#     def forward(self, x):
#         h, w = x.shape[-2:]
#
#         import pdb;
#         pdb.set_trace()
#
#         low = self.low_extractor(x)
#         low = self.low_proj(low)
#         low = self.eca_blocks(low)
#
#         prob_change, loss_lf = self.posterior_refiner(low)
#         gamma_feat = self.gamma_proj(prob_change)
#
#         if gamma_feat.shape[-2:] != (h, w):
#             gamma_feat = F.interpolate(
#                 gamma_feat, size=(h, w), mode='bilinear', align_corners=False
#             )
#
#         gamma_feat = self.out_proj(gamma_feat)
#
#         flow = x * (1.0 + self.lf_alpha * gamma_feat)
#
#
#         return flow, loss_lf
#
#
# # =========================================================
# # 3. Final Combined Module
# #    Ffus -> low branch / high branch in parallel
# #    then gated residual fusion
# # =========================================================
# class SPALPMFused_low(nn.Module):
#     """
#     最终组合模块：
#     - 高频增强：拉普拉斯金字塔
#     - 低频校正：DTCWT + ECA + 邻接 CRF
#     - 融合方式：以原始特征 x 为主干，flow / fhigh 作为门控残差项注入
#     """
#
#     def __init__(
#         self,
#         channels,
#         # high-frequency branch
#         levels=1,
#         beta=0.1,
#         gamma=0.1,
#         reduction=4,
#         hf_alpha=0.1,
#         # low-frequency branch
#         dtcwt_J=1,
#         eca_k_size=7,
#         num_eca_blocks=3,
#         lf_alpha=0.1,
#         crf_num_iters=10,
#         crf_hidden_ratio=0.75,
#         crf_sigma_feat=0.8,
#         crf_sigma_pos=0.8,
#         crf_max_hw=24
#     ):
#         super().__init__()
#
#         self.high_enhancer = SPALPMHighEnhancer(
#             channels=channels,
#             levels=levels,
#             beta=beta,
#             gamma=gamma,
#             reduction=reduction,
#             alpha=hf_alpha
#         )
#
#         self.low_refiner = DTCWTLowRefiner(
#             channels=channels,
#             dtcwt_J=dtcwt_J,
#             eca_k_size=eca_k_size,
#             num_eca_blocks=num_eca_blocks,
#             lf_alpha=lf_alpha,
#             crf_num_iters=crf_num_iters,
#             crf_hidden_ratio=crf_hidden_ratio,
#             crf_sigma_feat=crf_sigma_feat,
#             crf_sigma_pos=crf_sigma_pos,
#             crf_max_hw=crf_max_hw
#         )
#
#         self.low_proj = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channels)
#         )
#
#         self.high_proj = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channels)
#         )
#
#         self.low_gate = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=1, bias=True),
#             nn.Sigmoid()
#         )
#
#         self.high_gate = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=1, bias=True),
#             nn.Sigmoid()
#         )
#
#         self.out_proj = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         """
#         输入:
#             x = Ffus
#
#         输出:
#             fused_out: 最终融合特征
#             loss_sp:   高频结构保持损失
#             loss_lf:   低频CRF校正损失
#         """
#         flow, loss_lf = self.low_refiner(x)
#         fhigh, loss_sp = self.high_enhancer(x)
#
#         delta_l = self.low_proj(flow)
#         delta_h = self.high_proj(fhigh)
#
#         gl = self.low_gate(torch.cat([x, flow], dim=1))
#         gh = self.high_gate(torch.cat([x, fhigh], dim=1))
#
#         fused_out = x + gl * delta_l + gh * delta_h
#         fused_out = self.out_proj(fused_out)
#
#         return fused_out, loss_sp, loss_lf

import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_wavelets import DTCWTForward


# =========================================================
# 0. Visualization Utilities
# =========================================================
def _is_main_process():
    """
    DDP 多卡训练时只让 rank 0 保存图片，避免多卡重复写文件。
    单卡训练时默认返回 True。
    """
    rank = int(os.environ.get("RANK", "0"))
    return rank == 0


def _tensor_to_gray_image(t, use_abs=False, eps=1e-6):
    """
    将特征张量转换为可保存的灰度图。

    支持:
        [B, C, H, W]
        [C, H, W]
        [H, W]

    默认保存 batch 中第 1 张，并对通道求均值。
    use_abs=True 适合可视化拉普拉斯残差这类正负高频响应。
    """
    with torch.no_grad():
        t = t.detach().float().cpu()

        if t.ndim == 4:
            t = t[0]

        if t.ndim == 3:
            if use_abs:
                t = t.abs()
            t = t.mean(dim=0)
        elif t.ndim == 2:
            if use_abs:
                t = t.abs()
        else:
            raise ValueError(f"Unsupported tensor shape: {t.shape}")

        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

        t_min = t.min()
        t_max = t.max()

        if (t_max - t_min) < eps:
            img = torch.zeros_like(t)
        else:
            img = (t - t_min) / (t_max - t_min + eps)

        img = (img * 255.0).clamp(0, 255).byte().numpy()
        return img


def _save_gray_tensor(t, save_path, use_abs=False):
    """
    保存单张灰度特征图。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img = _tensor_to_gray_image(t, use_abs=use_abs)
    Image.fromarray(img).save(str(save_path))


def _tensor_to_phase_image(t, eps=1e-6):
    """
    将相位张量保存为灰度图。

    输入相位默认位于 [-pi, pi]，这里采用固定映射：
        -pi -> 0
         0  -> 127
         pi -> 255

    相比 min-max 归一化，固定范围映射更适合比较不同 step 的相位可视化结果。
    """
    with torch.no_grad():
        t = t.detach().float().cpu()

        if t.ndim == 4:
            t = t[0]

        if t.ndim == 3:
            t = t.mean(dim=0)
        elif t.ndim != 2:
            raise ValueError(f"Unsupported phase tensor shape: {t.shape}")

        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        t = (t + math.pi) / (2.0 * math.pi + eps)
        img = (t * 255.0).clamp(0, 255).byte().numpy()
        return img


def _save_phase_tensor(t, save_path):
    """
    保存中心化后的相位图。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img = _tensor_to_phase_image(t)
    Image.fromarray(img).save(str(save_path))


# =========================================================
# 1. High-Frequency Enhancement Branch
# =========================================================
class GaussianPyramid(nn.Module):
    def __init__(self):
        super().__init__()

        kernel = torch.tensor(
            [[1., 4., 6., 4., 1.],
             [4., 16., 24., 16., 4.],
             [6., 24., 36., 24., 6.],
             [4., 16., 24., 16., 4.],
             [1., 4., 6., 4., 1.]]
        )

        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None, :, :])

    def gaussian_blur(self, x):
        c = x.shape[1]
        kernel = self.kernel.repeat(c, 1, 1, 1)
        return F.conv2d(x, kernel, padding=2, groups=c)

    def down(self, x):
        return F.interpolate(
            x,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=False
        )

    def up(self, x, size):
        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False
        )

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
    def __init__(self, channels, reduction=4, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = channels

        hidden = max(channels // reduction, 4)

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class FrequencyPrompt(nn.Module):
    def __init__(self, channels, reduction=4, eps=1e-6):
        super().__init__()

        self.eps = eps

        # 幅度分支仍使用 log(1 + |FFT|)，用于压缩频谱动态范围。
        self.amp_gate = ChannelGate(
            channels=channels,
            reduction=reduction,
            out_channels=channels
        )

        # 相位分支不再直接使用 atan2(angle) 参与训练。
        # 改为使用 cos(theta)=real/mag 与 sin(theta)=imag/mag。
        # 这样可以避免 phase 在 [-pi, pi] 周期边界处跳变，也能缓解幅度接近 0 时的梯度不稳定。
        self.phase_gate = ChannelGate(
            channels=channels * 2,
            reduction=reduction,
            out_channels=channels
        )

    @staticmethod
    def _safe_fft2(x):
        """
        计算 2D FFT。
        部分环境下 cuFFT 可能偶发报错，因此保留 CPU fallback。
        """
        try:
            return torch.fft.fft2(x, norm="ortho")
        except RuntimeError as e:
            if "cuFFT" in str(e) or "CUFFT" in str(e):
                x_cpu = x.detach().cpu()
                return torch.fft.fft2(x_cpu, norm="ortho").to(x.device)
            raise e

    def forward(self, x, return_raw=False):
        """
        return_raw=True 时额外返回用于可视化的中心化幅度谱和中心化相位谱。

        训练中实际使用：
            amp_gate_input   = log(1 + |FFT(x)|)
            phase_gate_input = [cos(theta), sin(theta)]

        可视化中保存：
            amp_vis   = log(1 + |fftshift(FFT(x))|)
            phase_vis = angle(fftshift(FFT(x)))
        """
        x = x.contiguous()
        orig_dtype = x.dtype

        # torch.fft 对 fp16/bf16 支持不如 fp32 稳定，这里统一转 fp32 做频域分解。
        x_fft = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

        fft = self._safe_fft2(x_fft)

        real = fft.real
        imag = fft.imag

        mag = torch.sqrt(real.pow(2) + imag.pow(2) + self.eps)
        amp_feat = torch.log1p(mag)

        cos_phase = real / mag
        sin_phase = imag / mag
        phase_feat = torch.cat([cos_phase, sin_phase], dim=1)

        amp_feat = torch.nan_to_num(amp_feat, nan=0.0, posinf=0.0, neginf=0.0)
        phase_feat = torch.nan_to_num(phase_feat, nan=0.0, posinf=0.0, neginf=0.0)

        amp_feat = amp_feat.to(orig_dtype)
        phase_feat = phase_feat.to(orig_dtype)

        m_amp = self.amp_gate(amp_feat)
        m_phase = self.phase_gate(phase_feat)

        if return_raw:
            # 注意：fftshift 只用于保存图片，不能代表训练时必须 shift。
            # 对卷积网络来说，训练中的频域特征只经过全局池化生成通道门控，shift 与否不影响全局均值；
            # 但可视化时 shift 后更符合人眼直觉，低频会位于图像中心。
            fft_shift = torch.fft.fftshift(fft.detach(), dim=(-2, -1))
            amp_vis = torch.log1p(torch.abs(fft_shift))
            phase_vis = torch.angle(fft_shift)

            amp_vis = torch.nan_to_num(amp_vis, nan=0.0, posinf=0.0, neginf=0.0).to(orig_dtype)
            phase_vis = torch.nan_to_num(phase_vis, nan=0.0, posinf=0.0, neginf=0.0).to(orig_dtype)

            return m_amp, m_phase, amp_vis, phase_vis

        return m_amp, m_phase


class SPALPMHighEnhancer(nn.Module):
    """
    高频增强支路：
    - 拉普拉斯金字塔分解
    - 幅度/相位门控
    - 残差式结构保持增强
    - 可选保存中间可视化结果
    """
    def __init__(
        self,
        channels,
        levels=4,
        beta=0.1,
        gamma=0.1,
        reduction=4,
        alpha=0.1,
        save_vis=True,
        vis_dir="/home/f523/disk2/nym/open-cd-main/vis_high_frequency",
        vis_interval=100,
        vis_max_levels=None
    ):
        super().__init__()

        self.levels = int(levels)
        self.beta = beta
        self.gamma = gamma

        self.save_vis = save_vis
        self.vis_dir = Path(vis_dir)
        self.vis_interval = int(vis_interval)
        self.vis_max_levels = vis_max_levels
        self.vis_step = 0

        self.pyramid = GaussianPyramid()
        self.freq_prompt = FrequencyPrompt(
            channels=channels,
            reduction=reduction
        )

        self.amp_align = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            for _ in range(self.levels)
        ])

        self.phase_align = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            for _ in range(self.levels)
        ])

        self.out_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True
        )

        alpha = float(alpha)
        alpha = min(max(alpha, 1e-4), 1 - 1e-4)
        alpha_logit = math.log(alpha / (1 - alpha))
        self.alpha_logit = nn.Parameter(
            torch.tensor(alpha_logit, dtype=torch.float32)
        )

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def _save_high_vis(self, p, p_tilde, amp, phase):
        """
        保存高频增强过程中的可视化结果：
            1. 各层原始拉普拉斯残差
            2. 幅度谱特征
            3. 相位谱特征
            4. 各层增强后的拉普拉斯残差
        """
        if not self.save_vis:
            return

        if not _is_main_process():
            return

        if self.vis_interval <= 0:
            return

        if self.vis_step % self.vis_interval != 0:
            return

        step_dir = self.vis_dir / f"step_{self.vis_step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        _save_gray_tensor(
            amp,
            step_dir / "frequency_amplitude.png",
            use_abs=False
        )

        _save_phase_tensor(
            phase,
            step_dir / "frequency_phase.png"
        )

        max_levels = len(p)
        if self.vis_max_levels is not None:
            max_levels = min(max_levels, int(self.vis_max_levels))

        for l in range(max_levels):
            _save_gray_tensor(
                p[l],
                step_dir / f"laplacian_residual_level_{l}_before.png",
                use_abs=True
            )

            _save_gray_tensor(
                p_tilde[l],
                step_dir / f"laplacian_residual_level_{l}_after.png",
                use_abs=True
            )

    def reconstruct(self, g_last, p_tilde):
        cur = g_last

        for l in reversed(range(self.levels)):
            cur = p_tilde[l] + F.interpolate(
                cur,
                size=p_tilde[l].shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        return cur

    def forward(self, x):
        g, p = self.pyramid.build_laplacian_pyramid(
            x,
            levels=self.levels
        )

        m_amp, m_phase, amp, phase = self.freq_prompt(
            x,
            return_raw=True
        )

        p_tilde = []
        loss_sp = x.new_zeros(())

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

        self._save_high_vis(
            p=p,
            p_tilde=p_tilde,
            amp=amp,
            phase=phase
        )

        out_rec = self.reconstruct(g[-1], p_tilde)

        high_feat = x + self.alpha * self.out_proj(out_rec - x)

        self.vis_step += 1

        return high_feat, loss_sp


# =========================================================
# 2. Low-Frequency Calibration Branch
#    DTCWT low-frequency extraction + ECA attention + adjacency CRF
# =========================================================
class DTCWTLowFrequencyExtractor(nn.Module):
    def __init__(
        self,
        J=1,
        biort="near_sym_a",
        qshift="qshift_a"
    ):
        super().__init__()

        self.xfm = DTCWTForward(
            J=J,
            biort=biort,
            qshift=qshift
        )

    def forward(self, x):
        yl, yh = self.xfm(x)
        return yl


class ECALayer(nn.Module):
    """
    ECA-Net channel attention.
    """
    def __init__(self, channels, k_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                    # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)     # [B, 1, C]
        y = self.conv(y)                        # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)   # [B, C, 1, 1]
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ECAResidualBlock(nn.Module):
    """
    局部卷积 + ECA 通道重标定 + 残差连接。
    """
    def __init__(self, channels, k_size=9):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels)
        )

        self.eca = ECALayer(
            channels=channels,
            k_size=k_size
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.eca(out)
        out = out + identity
        out = self.act(out)

        return out


class AdjCRFPosteriorRefiner(nn.Module):
    """
    在池化后的低频特征图上构建 8 邻域 CRF，
    使用 mean-field 风格近似推断后验概率，
    输出变化类概率图 prob_change。
    """
    def __init__(
        self,
        channels,
        hidden_ratio=0.25,
        num_iters=5,
        sigma_feat=0.5,
        sigma_pos=0.3,
        max_hw=24
    ):
        super().__init__()

        hidden = max(int(channels * hidden_ratio), 8)

        self.num_iters = num_iters
        self.sigma_feat = sigma_feat
        self.sigma_pos = sigma_pos
        self.max_hw = max_hw

        self.embed_proj = nn.Conv2d(
            channels,
            hidden,
            kernel_size=1,
            bias=True
        )

        self.unary_proj = nn.Conv2d(
            channels,
            hidden,
            kernel_size=1,
            bias=True
        )

        self.classifier = nn.Conv2d(
            hidden,
            2,
            kernel_size=1,
            bias=True
        )

    @staticmethod
    def _build_coords(h, w, device, dtype):
        ys = torch.linspace(
            -1.0,
            1.0,
            steps=h,
            device=device,
            dtype=dtype
        )

        xs = torch.linspace(
            -1.0,
            1.0,
            steps=w,
            device=device,
            dtype=dtype
        )

        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy, xx], dim=0).unsqueeze(0)

        return coords

    @staticmethod
    def _shift(x, dy, dx):
        b, c, h, w = x.shape

        pad_l = max(dx, 0)
        pad_r = max(-dx, 0)
        pad_t = max(dy, 0)
        pad_b = max(-dy, 0)

        x_pad = F.pad(
            x,
            (pad_l, pad_r, pad_t, pad_b),
            mode="constant",
            value=0.0
        )

        y1 = pad_b
        y2 = y1 + h
        x1 = pad_r
        x2 = x1 + w

        return x_pad[:, :, y1:y2, x1:x2]

    @staticmethod
    def _shift_mask(h, w, dy, dx, device, dtype):
        mask = torch.ones(
            1,
            1,
            h,
            w,
            device=device,
            dtype=dtype
        )

        return AdjCRFPosteriorRefiner._shift(mask, dy, dx)

    def _pairwise_kernel(self, feat, coords):
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        weights = []

        for dy, dx in offsets:
            feat_n = self._shift(feat, dy, dx)
            coord_n = self._shift(coords, dy, dx)

            d_feat = ((feat - feat_n) ** 2).sum(
                dim=1,
                keepdim=True
            )

            d_pos = ((coords - coord_n) ** 2).sum(
                dim=1,
                keepdim=True
            )

            # 注意：
            # 当前实现没有显式使用 sigma_feat 和 sigma_pos。
            # 如果需要严格对应论文公式，可改为：
            # d_feat / (2 * self.sigma_feat ** 2) + d_pos / (2 * self.sigma_pos ** 2)
            w = torch.exp(-d_feat - d_pos)

            mask = self._shift_mask(
                feat.shape[-2],
                feat.shape[-1],
                dy,
                dx,
                feat.device,
                feat.dtype
            )

            w = w * mask
            weights.append((dy, dx, w))

        return weights

    def _mean_field_inference(self, unary_logits, pairwise_weights):
        q = torch.softmax(unary_logits, dim=1)

        for _ in range(self.num_iters):
            msg = 0.0

            for dy, dx, w in pairwise_weights:
                q_n = self._shift(q, dy, dx)
                msg = msg + w * q_n

            q = torch.softmax(unary_logits + msg, dim=1)

        return q

    def forward(self, x):
        b, c, h, w = x.shape

        hs = min(h, self.max_hw)
        ws = min(w, self.max_hw)

        if hs != h or ws != w:
            x_small = F.adaptive_avg_pool2d(x, (hs, ws))
        else:
            x_small = x

        feat = self.embed_proj(x_small)
        unary_feat = self.unary_proj(x_small)
        unary_logits = self.classifier(unary_feat)

        coords = self._build_coords(
            hs,
            ws,
            x_small.device,
            x_small.dtype
        ).expand(b, -1, -1, -1)

        pairwise_weights = self._pairwise_kernel(
            feat,
            coords
        )

        post_q = self._mean_field_inference(
            unary_logits,
            pairwise_weights
        )

        prob_change_small = post_q[:, 1:2, :, :]

        init_prob = torch.softmax(unary_logits, dim=1)[:, 1:2, :, :]
        loss_lf = F.l1_loss(
            prob_change_small,
            init_prob.detach()
        )

        if hs != h or ws != w:
            prob_change = F.interpolate(
                prob_change_small,
                size=(h, w),
                mode="bilinear",
                align_corners=False
            )
        else:
            prob_change = prob_change_small

        return prob_change, loss_lf


class DTCWTLowRefiner(nn.Module):
    """
    低频校正支路：
        Ffus -> DTCWT低频 -> ECA attention -> 邻接CRF
             -> gamma map -> 通道映射 -> 低频校正结果 Flow
    """
    def __init__(
        self,
        channels,
        dtcwt_J=1,
        biort="near_sym_a",
        qshift="qshift_a",
        eca_k_size=3,
        num_eca_blocks=1,
        lf_alpha=0.1,
        crf_hidden_ratio=0.25,
        crf_num_iters=5,
        crf_sigma_feat=0.5,
        crf_sigma_pos=0.3,
        crf_max_hw=24
    ):
        super().__init__()

        self.low_extractor = DTCWTLowFrequencyExtractor(
            J=dtcwt_J,
            biort=biort,
            qshift=qshift
        )

        self.low_proj = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.eca_blocks = nn.Sequential(*[
            ECAResidualBlock(
                channels=channels,
                k_size=eca_k_size
            )
            for _ in range(num_eca_blocks)
        ])

        self.posterior_refiner = AdjCRFPosteriorRefiner(
            channels=channels,
            hidden_ratio=crf_hidden_ratio,
            num_iters=crf_num_iters,
            sigma_feat=crf_sigma_feat,
            sigma_pos=crf_sigma_pos,
            max_hw=crf_max_hw
        )

        self.gamma_proj = nn.Sequential(
            nn.Conv2d(
                1,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        lf_alpha = float(lf_alpha)
        lf_alpha = min(max(lf_alpha, 1e-4), 1 - 1e-4)
        lf_alpha_logit = math.log(lf_alpha / (1 - lf_alpha))

        self.lf_alpha_logit = nn.Parameter(
            torch.tensor(lf_alpha_logit, dtype=torch.float32)
        )

    @property
    def lf_alpha(self):
        return torch.sigmoid(self.lf_alpha_logit)

    def forward(self, x):
        h, w = x.shape[-2:]

        low = self.low_extractor(x)
        low = self.low_proj(low)
        low = self.eca_blocks(low)

        prob_change, loss_lf = self.posterior_refiner(low)
        gamma_feat = self.gamma_proj(prob_change)

        if gamma_feat.shape[-2:] != (h, w):
            gamma_feat = F.interpolate(
                gamma_feat,
                size=(h, w),
                mode="bilinear",
                align_corners=False
            )

        gamma_feat = self.out_proj(gamma_feat)

        flow = x * (1.0 + self.lf_alpha * gamma_feat)

        return flow, loss_lf


# =========================================================
# 3. Final Combined Module
#    Ffus -> low branch / high branch in parallel
#    then gated residual fusion
# =========================================================
class SPALPMFused_low(nn.Module):
    """
    最终组合模块：
    - 高频增强：拉普拉斯金字塔
    - 低频校正：DTCWT + ECA + 邻接 CRF
    - 融合方式：以原始特征 x 为主干，flow / fhigh 作为门控残差项注入
    """

    def __init__(
        self,
        channels,
        # high-frequency branch
        levels=4,
        beta=0.1,
        gamma=0.1,
        reduction=4,
        hf_alpha=0.1,
        save_hf_vis=True,
        hf_vis_dir="/home/f523/disk2/nym/open-cd-main/vis_high_frequency",
        hf_vis_interval=100,
        hf_vis_max_levels=None,
        # low-frequency branch
        dtcwt_J=1,
        eca_k_size=7,
        num_eca_blocks=3,
        lf_alpha=0.1,
        crf_num_iters=10,
        crf_hidden_ratio=0.75,
        crf_sigma_feat=0.8,
        crf_sigma_pos=0.8,
        crf_max_hw=24
    ):
        super().__init__()

        self.high_enhancer = SPALPMHighEnhancer(
            channels=channels,
            levels=levels,
            beta=beta,
            gamma=gamma,
            reduction=reduction,
            alpha=hf_alpha,
            save_vis=save_hf_vis,
            vis_dir=hf_vis_dir,
            vis_interval=hf_vis_interval,
            vis_max_levels=hf_vis_max_levels
        )

        self.low_refiner = DTCWTLowRefiner(
            channels=channels,
            dtcwt_J=dtcwt_J,
            eca_k_size=eca_k_size,
            num_eca_blocks=num_eca_blocks,
            lf_alpha=lf_alpha,
            crf_num_iters=crf_num_iters,
            crf_hidden_ratio=crf_hidden_ratio,
            crf_sigma_feat=crf_sigma_feat,
            crf_sigma_pos=crf_sigma_pos,
            crf_max_hw=crf_max_hw
        )

        self.low_proj = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels)
        )

        self.high_proj = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels)
        )

        self.low_gate = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.Sigmoid()
        )

        self.high_gate = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.Sigmoid()
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        输入:
            x = Ffus

        输出:
            fused_out: 最终融合特征
            loss_sp:   高频结构保持损失
            loss_lf:   低频CRF校正损失
        """
        ###a+b+c
        flow, loss_lf = self.low_refiner(x)
        fhigh, loss_sp = self.high_enhancer(x)

        delta_l = self.low_proj(flow)
        delta_h = self.high_proj(fhigh)

        gl = self.low_gate(
            torch.cat([x, flow], dim=1)
        )

        gh = self.high_gate(
            torch.cat([x, fhigh], dim=1)
        )

        fused_out = x + gl * delta_l + gh * delta_h
        fused_out = self.out_proj(fused_out)

        return fused_out, loss_sp, loss_lf


        ###无c
        # flow, loss_lf = self.low_refiner(x)
        # loss_sp = x.new_zeros(())
        #
        # delta_l = self.low_proj(flow)
        #
        # gl = self.low_gate(
        #     torch.cat([x, flow], dim=1)
        # )
        #
        # fused_out = x + gl * delta_l
        # fused_out = self.out_proj(fused_out)
        #
        # return fused_out, loss_sp, loss_lf


        ###无bc
        # loss_sp = x.new_zeros(())
        # loss_lf = x.new_zeros(())
        #
        # fused_out = self.out_proj(x)
        #
        # return fused_out, loss_sp, loss_lf


# =========================================================
# 4. Simple Running Test
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(2, 256, 64, 64).to(device)

    module = SPALPMFused_low(
        channels=256,
        levels=4,
        save_hf_vis=True,
        hf_vis_dir="/home/f523/disk2/nym/open-cd-main/debug_vis/high_frequency",
        hf_vis_interval=1,
        hf_vis_max_levels=4
    ).to(device)

    module.train()

    fused_out, loss_sp, loss_lf = module(x)

    print("input:", x.shape)
    print("fused_out:", fused_out.shape)
    print("loss_sp:", loss_sp.item())
    print("loss_lf:", loss_lf.item())
    print("visualization saved to: /home/f523/disk2/nym/open-cd-main/debug_vis/high_frequency")