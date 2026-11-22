import torch
import torch.nn as nn
from mmseg.models.decode_heads.segformer_head import SegformerHead
from opencd.registry import MODELS
from mmseg.models.utils import resize

from .spalpm_fused_lowpro import SPALPMFused_low


@MODELS.register_module()
class SPALPMFusedSegformerHead_low(SegformerHead):
    """
    在四阶段特征完成上采样、拼接，并由 fusion_conv 压到 c 通道后，
    同时进行：
    1) 高频增强（拉普拉斯金字塔）
    2) 低频校正（DTCWT + FFC + 邻接CRF）
    最终将 [Ffus, Flow, Fhigh] 拼接融合。
    """

    def __init__(self,
                 spalpm_levels=1,
                 spalpm_beta=0.1,
                 spalpm_gamma=0.1,
                 spalpm_reduction=4,
                 hf_alpha=0.1,
                 dtcwt_J=1,
                 lf_alpha=0.1,
                 crf_num_iters=5,
                 crf_hidden_ratio=0.25,
                 crf_sigma_feat=0.5,
                 crf_sigma_pos=0.3,
                 crf_max_hw=24,
                 eca_k_size=3,
                 **kwargs):
        super().__init__(**kwargs)

        self.spalpm = SPALPMFused_low(
            channels=self.channels,

            # high-frequency branch
            levels=spalpm_levels,
            beta=spalpm_beta,
            gamma=spalpm_gamma,
            reduction=spalpm_reduction,
            hf_alpha=hf_alpha,

            # low-frequency branch
            dtcwt_J=dtcwt_J,
            lf_alpha=lf_alpha,
            crf_num_iters=crf_num_iters,
            crf_hidden_ratio=crf_hidden_ratio,
            crf_sigma_feat=crf_sigma_feat,
            crf_sigma_pos=crf_sigma_pos,
            crf_max_hw=crf_max_hw,
            eca_k_size = eca_k_size,
        )

        self._loss_sp = None
        self._loss_lf = None

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            )

        # 原始融合特征 Ffus
        fused = self.fusion_conv(torch.cat(outs, dim=1))

        # 同时做高频增强 + 低频校正 + 拼接融合
        fused_refined, loss_sp, loss_lf = self.spalpm(fused)

        self._loss_sp = loss_sp
        self._loss_lf = loss_lf

        out = self.cls_seg(fused_refined)

        return out