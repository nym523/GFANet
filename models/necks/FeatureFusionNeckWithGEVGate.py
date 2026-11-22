import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from opencd.registry import MODELS


@MODELS.register_module()
class FeatureFusionNeckWithGEVGate(BaseModule):

    def __init__(self,
                 policy='abs_diff',
                 in_channels=(64, 128, 320, 512),
                 out_indices=(0, 1, 2, 3),
                 stat_root='',
                 stat_channels=32):
        super().__init__()

        self.policy = policy
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.stat_root = stat_root
        self.stat_channels = stat_channels

        self.stat_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, stat_channels, 3, 1, 1),
            nn.BatchNorm2d(stat_channels),
            nn.ReLU(inplace=True),
        )

        self.stat_projs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        for c in self.in_channels:
            feat_c = c * 2 if self.policy == 'concat' else c
            self.stat_projs.append(nn.Conv2d(stat_channels, feat_c, 1))
            self.gate_convs.append(nn.Conv2d(feat_c, feat_c, 1))

    @staticmethod
    def fusion(x1, x2, policy):
        if policy == 'concat':
            return torch.cat([x1, x2], dim=1)
        elif policy == 'sum':
            return x1 + x2
        elif policy == 'diff':
            return x2 - x1
        elif policy == 'abs_diff':
            return torch.abs(x1 - x2)
        else:
            raise ValueError(f'Unsupported policy: {policy}')

    # def _extract_img_stem(self, meta):
    #     for k in ['img_path', 'img_path_from', 'filename', 'ori_filename']:
    #         if k in meta:
    #             return Path(meta[k]).stem
    #     raise KeyError(f'Cannot find image path key in meta: {meta.keys()}')

    def _extract_img_stem(self, meta):

        print('meta keys =', meta.keys())
        for k, v in meta.items():
            print(k, type(v), v if isinstance(v, (str, list, tuple)) else '...')


        # 按优先级找
        candidate_keys = [
            'img_path_from',
            'img_path',
            'img_path_to',
            'filename',
            'ori_filename',
        ]

        for k in candidate_keys:
            if k not in meta:
                continue

            v = meta[k]
            if v is None:
                continue

            if isinstance(v, (list, tuple)):
                if len(v) == 0:
                    continue
                v = v[0]

            if isinstance(v, (str, bytes)):
                return Path(v).stem

            try:
                return Path(v).stem
            except Exception:
                pass

        raise KeyError(
            f'Failed to extract image stem. meta keys = {list(meta.keys())}, '
            f'meta = {meta}'
        )

    def _guess_npz_path(self, stem):
        return os.path.join(self.stat_root, stem, 'region_gev_lrt_results.npz')

    def _read_stat_from_npz(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        if 'd_map' not in data:
            raise KeyError(f"'d_map' not found in {npz_path}, available keys: {data.files}")

        stat_map = data['d_map'].astype(np.float32)
        stat_map = np.nan_to_num(stat_map, nan=0.0, posinf=0.0, neginf=0.0)

        # 推荐：log压缩 + 归一化
        stat_map = np.log1p(stat_map)
        stat_map = (stat_map - stat_map.min()) / (stat_map.max() - stat_map.min() + 1e-6)

        return stat_map

    def _load_batch_stat_map(self, batch_img_metas, device, dtype):
        stat_list = []

        for meta in batch_img_metas:
            stem = self._extract_img_stem(meta)
            npz_path = self._guess_npz_path(stem)

            if not os.path.exists(npz_path):
                raise FileNotFoundError(f'Statistical map not found: {npz_path}')

            stat_map = self._read_stat_from_npz(npz_path)
            stat_list.append(stat_map)

        stat_arr = np.stack(stat_list, axis=0)  # [B,H,W]
        stat_tensor = torch.from_numpy(stat_arr).to(device=device, dtype=dtype).unsqueeze(1)
        return stat_tensor

    def forward(self, x1, x2, batch_img_metas=None):
        assert len(x1) == len(x2)

        outs = []
        for i in range(len(x1)):
            outs.append(self.fusion(x1[i], x2[i], self.policy))

        outs = [outs[i] for i in self.out_indices]

        if batch_img_metas is None:
            return tuple(outs)

        stat_map = self._load_batch_stat_map(
            batch_img_metas,
            device=outs[0].device,
            dtype=outs[0].dtype
        )  # [B,1,H,W]

        stat_feat0 = self.stat_encoder(stat_map)

        fused_outs = []
        for idx, feat in enumerate(outs):
            stat_l = F.interpolate(
                stat_feat0,
                size=feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            stat_l = self.stat_projs[idx](stat_l)
            gate = torch.sigmoid(self.gate_convs[idx](stat_l))

            feat = feat * (1.0 + 1.0*gate)
            fused_outs.append(feat)

        return tuple(fused_outs)

# import os
# import numpy as np
# from pathlib import Path
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmengine.model import BaseModule
# from opencd.registry import MODELS


# @MODELS.register_module()
# class FeatureFusionNeckWithGEVGate(BaseModule):
#
#     def __init__(self,
#                  policy='abs_diff',
#                  in_channels=(64, 128, 320, 512),
#                  out_indices=(0, 1, 2, 3),
#                  stat_root='',
#                  stat_channels=32):
#         super().__init__()
#
#         self.policy = policy
#         self.in_channels = in_channels
#         self.out_indices = out_indices
#         self.stat_root = stat_root
#         self.stat_channels = stat_channels
#
#         self.stat_encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, 1, 1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(16, stat_channels, 3, 1, 1),
#             nn.BatchNorm2d(stat_channels),
#             nn.ReLU(inplace=True),
#         )
#
#         self.stat_projs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#         self.gate_scales = nn.ParameterList()   # 新增
#
#         for c in self.in_channels:
#             feat_c = c * 2 if self.policy == 'concat' else c
#             self.stat_projs.append(nn.Conv2d(stat_channels, feat_c, 1))
#             self.gate_convs.append(nn.Conv2d(feat_c, feat_c, 1))
#
#             # 每层一个可学习缩放系数
#             self.gate_scales.append(
#                 nn.Parameter(torch.tensor(0.6, dtype=torch.float32))
#             )
#
#     @staticmethod
#     def fusion(x1, x2, policy):
#         if policy == 'concat':
#             return torch.cat([x1, x2], dim=1)
#         elif policy == 'sum':
#             return x1 + x2
#         elif policy == 'diff':
#             return x2 - x1
#         elif policy == 'abs_diff':
#             return torch.abs(x1 - x2)
#         else:
#             raise ValueError(f'Unsupported policy: {policy}')
#
#     def _extract_img_stem(self, meta):
#
#         print('meta keys =', meta.keys())
#         for k, v in meta.items():
#             print(k, type(v), v if isinstance(v, (str, list, tuple)) else '...')
#
#         candidate_keys = [
#             'img_path_from',
#             'img_path',
#             'img_path_to',
#             'filename',
#             'ori_filename',
#         ]
#
#         for k in candidate_keys:
#             if k not in meta:
#                 continue
#
#             v = meta[k]
#             if v is None:
#                 continue
#
#             if isinstance(v, (list, tuple)):
#                 if len(v) == 0:
#                     continue
#                 v = v[0]
#
#             if isinstance(v, (str, bytes)):
#                 return Path(v).stem
#
#             try:
#                 return Path(v).stem
#             except Exception:
#                 pass
#
#         raise KeyError(
#             f'Failed to extract image stem. meta keys = {list(meta.keys())}, '
#             f'meta = {meta}'
#         )
#
#     def _guess_npz_path(self, stem):
#         return os.path.join(self.stat_root, stem, 'region_gev_lrt_results.npz')
#
#     def _read_stat_from_npz(self, npz_path):
#         data = np.load(npz_path, allow_pickle=True)
#         if 'd_map' not in data:
#             raise KeyError(f"'d_map' not found in {npz_path}, available keys: {data.files}")
#
#         stat_map = data['d_map'].astype(np.float32)
#         stat_map = np.nan_to_num(stat_map, nan=0.0, posinf=0.0, neginf=0.0)
#
#         stat_map = np.log1p(stat_map)
#         stat_map = (stat_map - stat_map.min()) / (stat_map.max() - stat_map.min() + 1e-6)
#
#         return stat_map
#
#     def _load_batch_stat_map(self, batch_img_metas, device, dtype):
#         stat_list = []
#
#         for meta in batch_img_metas:
#             stem = self._extract_img_stem(meta)
#             npz_path = self._guess_npz_path(stem)
#
#             if not os.path.exists(npz_path):
#                 raise FileNotFoundError(f'Statistical map not found: {npz_path}')
#
#             stat_map = self._read_stat_from_npz(npz_path)
#             stat_list.append(stat_map)
#
#         stat_arr = np.stack(stat_list, axis=0)
#         stat_tensor = torch.from_numpy(stat_arr).to(device=device, dtype=dtype).unsqueeze(1)
#         return stat_tensor
#
#     def forward(self, x1, x2, batch_img_metas=None):
#         assert len(x1) == len(x2)
#
#         outs = []
#         for i in range(len(x1)):
#             outs.append(self.fusion(x1[i], x2[i], self.policy))
#
#         outs = [outs[i] for i in self.out_indices]
#
#         if batch_img_metas is None:
#             return tuple(outs)
#
#         stat_map = self._load_batch_stat_map(
#             batch_img_metas,
#             device=outs[0].device,
#             dtype=outs[0].dtype
#         )
#
#         stat_feat0 = self.stat_encoder(stat_map)
#
#         fused_outs = []
#         for idx, feat in enumerate(outs):
#             stat_l = F.interpolate(
#                 stat_feat0,
#                 size=feat.shape[2:],
#                 mode='bilinear',
#                 align_corners=False
#             )
#             stat_l = self.stat_projs[idx](stat_l)
#             gate = torch.sigmoid(self.gate_convs[idx](stat_l))
#
#             scale = self.gate_scales[idx]   # 可学习参数
#             feat = feat * (1.0 + scale * gate)
#
#             fused_outs.append(feat)
#
#         return tuple(fused_outs)
# import os
# from pathlib import Path
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmengine.model import BaseModule
# from opencd.registry import MODELS
#
#
# @MODELS.register_module()
# class FeatureFusionNeckWithGEVGate(BaseModule):
#     """
#     在原 FeatureFusionNeckWithGEVGate 基础上加入以下小 trick：
#
#     1) gate 不再只由统计图生成，而是：
#        gate = stat_gate + feat_gate
#        即统计先验 + 当前融合特征 共同决定调制强度
#
#     2) 中心化 gate：
#        gate = 2 * sigmoid(logits / tau) - 1
#        取值范围 [-1, 1]
#        因此既能增强，也能抑制
#
#     3) 双路调制：
#        out = feat * (1 + scale * gate) + add_scale * add_term
#        其中 add_term 由统计图分支生成，用于给弱变化区域做加性补偿
#
#     4) 统计图编码分支默认使用 GroupNorm，增强小 batch 训练稳定性
#     """
#
#     def __init__(self,
#                  policy='abs_diff',
#                  in_channels=(64, 128, 320, 512),
#                  out_indices=(0, 1, 2, 3),
#                  stat_root='',
#                  stat_channels=32,
#                  stat_mid_channels=16,
#                  norm_groups=8,
#                  gate_init=0.2,
#                  add_init=0.1,
#                  tau=1.5,
#                  use_centered_gate=True,
#                  use_additive=True,
#                  use_edge_map=False):
#         super().__init__()
#
#         self.policy = policy
#         self.in_channels = in_channels
#         self.out_indices = out_indices
#         self.stat_root = stat_root
#         self.stat_channels = stat_channels
#         self.stat_mid_channels = stat_mid_channels
#         self.norm_groups = norm_groups
#         self.use_centered_gate = use_centered_gate
#         self.use_additive = use_additive
#         self.use_edge_map = use_edge_map
#
#         stat_in_ch = 2 if use_edge_map else 1
#
#         def make_gn(c):
#             g = min(norm_groups, c)
#             while c % g != 0 and g > 1:
#                 g -= 1
#             return nn.GroupNorm(g, c)
#
#         # 统计图编码器
#         self.stat_encoder = nn.Sequential(
#             nn.Conv2d(stat_in_ch, stat_mid_channels, 3, 1, 1, bias=False),
#             make_gn(stat_mid_channels),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(stat_mid_channels, stat_channels, 3, 1, 1, bias=False),
#             make_gn(stat_channels),
#             nn.ReLU(inplace=True),
#         )
#
#         # 每层独立投影与调制头
#         self.stat_projs = nn.ModuleList()       # stat_feat -> feat channels
#         self.stat_gate_convs = nn.ModuleList()  # stat branch -> gate logits
#         self.feat_gate_convs = nn.ModuleList()  # feat branch -> gate logits
#         self.add_convs = nn.ModuleList()        # stat branch -> additive term
#
#         self.gate_scales = nn.ParameterList()
#         self.add_scales = nn.ParameterList()
#         self.taus = nn.ParameterList()
#
#         for c in self.in_channels:
#             feat_c = c * 2 if self.policy == 'concat' else c
#
#             self.stat_projs.append(
#                 nn.Conv2d(stat_channels, feat_c, kernel_size=1, stride=1, padding=0)
#             )
#
#             # 统计图分支 gate
#             self.stat_gate_convs.append(
#                 nn.Sequential(
#                     nn.Conv2d(feat_c, feat_c, 3, 1, 1, groups=1, bias=False),
#                     make_gn(feat_c),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(feat_c, feat_c, 1, 1, 0, bias=True)
#                 )
#             )
#
#             # 当前融合特征分支 gate
#             self.feat_gate_convs.append(
#                 nn.Sequential(
#                     nn.Conv2d(feat_c, feat_c, 3, 1, 1, groups=1, bias=False),
#                     make_gn(feat_c),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(feat_c, feat_c, 1, 1, 0, bias=True)
#                 )
#             )
#
#             # 加性补偿分支
#             self.add_convs.append(
#                 nn.Sequential(
#                     nn.Conv2d(feat_c, feat_c, 3, 1, 1, bias=False),
#                     make_gn(feat_c),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(feat_c, feat_c, 1, 1, 0, bias=True)
#                 )
#             )
#
#             # 可学习缩放系数
#             self.gate_scales.append(
#                 nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
#             )
#             self.add_scales.append(
#                 nn.Parameter(torch.tensor(float(add_init), dtype=torch.float32))
#             )
#             self.taus.append(
#                 nn.Parameter(torch.tensor(float(tau), dtype=torch.float32))
#             )
#
#     @staticmethod
#     def fusion(x1, x2, policy):
#         if policy == 'concat':
#             return torch.cat([x1, x2], dim=1)
#         elif policy == 'sum':
#             return x1 + x2
#         elif policy == 'diff':
#             return x2 - x1
#         elif policy == 'abs_diff':
#             return torch.abs(x1 - x2)
#         else:
#             raise ValueError(f'Unsupported policy: {policy}')
#
#     def _extract_img_stem(self, meta):
#         candidate_keys = [
#             'img_path_from',
#             'img_path',
#             'img_path_to',
#             'filename',
#             'ori_filename',
#         ]
#
#         for k in candidate_keys:
#             if k not in meta:
#                 continue
#
#             v = meta[k]
#             if v is None:
#                 continue
#
#             if isinstance(v, (list, tuple)):
#                 if len(v) == 0:
#                     continue
#                 v = v[0]
#
#             if isinstance(v, (str, bytes)):
#                 return Path(v).stem
#
#             try:
#                 return Path(v).stem
#             except Exception:
#                 pass
#
#         raise KeyError(
#             f'Failed to extract image stem. meta keys = {list(meta.keys())}, meta = {meta}'
#         )
#
#     def _guess_npz_path(self, stem):
#         return os.path.join(self.stat_root, stem, 'region_gev_lrt_results.npz')
#
#     def _read_stat_from_npz(self, npz_path):
#         data = np.load(npz_path, allow_pickle=True)
#         if 'd_map' not in data:
#             raise KeyError(f"'d_map' not found in {npz_path}, available keys: {data.files}")
#
#         stat_map = data['d_map'].astype(np.float32)
#         stat_map = np.nan_to_num(stat_map, nan=0.0, posinf=0.0, neginf=0.0)
#
#         # 对统计图做 log 压缩 + 归一化
#         stat_map = np.log1p(stat_map)
#         stat_min = stat_map.min()
#         stat_max = stat_map.max()
#         stat_map = (stat_map - stat_min) / (stat_max - stat_min + 1e-6)
#
#         return stat_map
#
#     @staticmethod
#     def _build_edge_map(stat_map):
#         """
#         stat_map: [H, W], numpy.float32
#         用简单差分构造边缘图，避免额外依赖 cv2
#         """
#         gx = np.zeros_like(stat_map, dtype=np.float32)
#         gy = np.zeros_like(stat_map, dtype=np.float32)
#
#         gx[:, 1:] = np.abs(stat_map[:, 1:] - stat_map[:, :-1])
#         gy[1:, :] = np.abs(stat_map[1:, :] - stat_map[:-1, :])
#
#         edge = gx + gy
#         edge_min = edge.min()
#         edge_max = edge.max()
#         edge = (edge - edge_min) / (edge_max - edge_min + 1e-6)
#         return edge
#
#     def _load_batch_stat_map(self, batch_img_metas, device, dtype):
#         stat_list = []
#
#         for meta in batch_img_metas:
#             stem = self._extract_img_stem(meta)
#             npz_path = self._guess_npz_path(stem)
#
#             if not os.path.exists(npz_path):
#                 raise FileNotFoundError(f'Statistical map not found: {npz_path}')
#
#             stat_map = self._read_stat_from_npz(npz_path)
#
#             if self.use_edge_map:
#                 edge_map = self._build_edge_map(stat_map)
#                 stat_map = np.stack([stat_map, edge_map], axis=0)  # [2, H, W]
#             else:
#                 stat_map = stat_map[None, ...]  # [1, H, W]
#
#             stat_list.append(stat_map)
#
#         stat_arr = np.stack(stat_list, axis=0)  # [B, C, H, W]
#         stat_tensor = torch.from_numpy(stat_arr).to(device=device, dtype=dtype)
#         return stat_tensor
#
#     def _compute_gate(self, idx, feat, stat_l):
#         """
#         feat:   [B, C, H, W]
#         stat_l: [B, C, H, W]
#         """
#         stat_logits = self.stat_gate_convs[idx](stat_l)
#         feat_logits = self.feat_gate_convs[idx](feat)
#         logits = stat_logits + feat_logits
#
#         tau = torch.clamp(self.taus[idx], min=0.5, max=5.0)
#
#         if self.use_centered_gate:
#             gate = 2.0 * torch.sigmoid(logits / tau) - 1.0
#         else:
#             gate = torch.sigmoid(logits / tau)
#
#         return gate
#
#     def forward(self, x1, x2, batch_img_metas=None):
#         assert len(x1) == len(x2), 'x1 and x2 must have the same number of stages.'
#
#         # Step 1: 原始双时相融合
#         outs = []
#         for i in range(len(x1)):
#             outs.append(self.fusion(x1[i], x2[i], self.policy))
#
#         outs = [outs[i] for i in self.out_indices]
#
#         # 没有 meta 时，直接返回原融合结果
#         if batch_img_metas is None:
#             return tuple(outs)
#
#         # Step 2: 载入统计图
#         stat_map = self._load_batch_stat_map(
#             batch_img_metas,
#             device=outs[0].device,
#             dtype=outs[0].dtype
#         )
#
#         # Step 3: 统计图编码
#         stat_feat0 = self.stat_encoder(stat_map)
#
#         # Step 4: 分层调制
#         fused_outs = []
#         for idx, feat in enumerate(outs):
#             # 上采样到当前层空间尺寸
#             stat_l = F.interpolate(
#                 stat_feat0,
#                 size=feat.shape[2:],
#                 mode='bilinear',
#                 align_corners=False
#             )
#
#             # 投影到当前层通道数
#             stat_l = self.stat_projs[idx](stat_l)
#
#             # 联合 gate：统计图分支 + 特征分支
#             gate = self._compute_gate(idx, feat, stat_l)
#
#             # 乘性调制
#             scale = self.gate_scales[idx]
#             feat_mod = feat * (1.0 + scale * gate)
#
#             # 加性补偿
#             if self.use_additive:
#                 add_term = self.add_convs[idx](stat_l)
#                 add_scale = self.add_scales[idx]
#                 feat_mod = feat_mod + add_scale * add_term
#
#             fused_outs.append(feat_mod)
#
#         return tuple(fused_outs)