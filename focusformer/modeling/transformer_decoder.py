# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.events import get_event_storage
from detectron2.utils import comm
from .head import TRANSFORMER_DECODER_REGISTRY

from .position_encoding import PositionEmbeddingSine3D

class FocusFeatureSampler(nn.Module):
    def __init__(self, feature_channels, hidden_dim):
        super().__init__()
        # 如果 Pixel Decoder 输出通道数 != Transformer hidden_dim，需要投影
        if feature_channels != hidden_dim:
            self.proj = nn.Linear(feature_channels, hidden_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, feature_maps, proposal_centers):
        """
        Args:
            feature_maps: [B, C, Z, H, W] (来自 Pixel Decoder 的高分辨特征)
            proposal_centers: [B, N, 3] (来自 Data Mapper, 范围 0-1, 顺序 x, y, z)
        Returns:
            sampled_queries: [B, N, hidden_dim]
        """
        # 1. 检查输入维度
        # 如果 feature_maps 是 [B, C, H, W] (2D)，需要 unsqueeze Z 轴 (假设 Z=1)
        # 但既然你做 Z-stack，这里假设 feature_maps 已经是 5D Tensor
        if feature_maps.dim() == 4:
             # Fallback for 2D debugging
             feature_maps = feature_maps.unsqueeze(2) 

        B, C, Z, H, W = feature_maps.shape
        N = proposal_centers.shape[1]
        
        # 2. 坐标变换: [0, 1] -> [-1, 1] (Grid Sample 标准)
        # proposal_centers is (x, y, z)
        # grid_sample expects (x, y, z) for 5D input
        grid = proposal_centers * 2.0 - 1.0
        grid = grid.view(B, N, 1, 1, 3) # [B, N, 1, 1, 3]
        
        # 3. 特征采样 (Bilinear Interpolation)
        # padding_mode='zeros' 保证如果在边界外采样为 0
        sampled_feats = F.grid_sample(
            feature_maps, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        ) # Output: [B, C, N, 1, 1]
        
        # 4. 调整维度 [B, C, N, 1, 1] -> [B, N, C]
        sampled_feats = sampled_feats.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        
        # 5. 投影到 hidden_dim
        init_content = self.proj(sampled_feats)
        
        return init_content

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

class SpatialGate(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=3):
        super().__init__()
        """
        一个“折 Z 维到 batch，用 2D Conv 做空间门控”的实现：
        - 输入:  [B, C, Z, H, W]
        - 在每个 Z-slice 上做 2D 空间注意力 (Conv2d)
        - 输出:  [B, 1, Z, H, W]
        这样能保留 XY 上的感受野，同时利用高度优化的 Conv2d，显著快于 3D Conv3d。
        """
        assert channels % reduction == 0, "channels 必须能被 reduction 整除"
        mid_channels = channels // reduction

        padding = kernel_size // 2
        # 避免 mid_channels 太小时 GroupNorm 报错
        num_groups = min(4, mid_channels)

        self.gate_net_2d = nn.Sequential(
            # 1. 1x1 Conv 降维
            nn.Conv2d(channels, mid_channels, kernel_size=1),
            nn.GroupNorm(num_groups, mid_channels),
            nn.ReLU(),

            # 2. Depthwise kxk Conv, 仅在 XY 上卷积
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=mid_channels,  # Depthwise
            ),
            nn.ReLU(),

            # 3. 1x1 Conv 输出单通道 Gate
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, Z, H, W]
        Returns:
            gate: [B, 1, Z, H, W]
        """
        B, C, Z, H, W = x.shape

        # [B, C, Z, H, W] -> [B*Z, C, H, W]
        x_2d = x.contiguous().view(B * Z, C, H, W)

        # 在每个 Z-slice 上做 2D Spatial Gate
        gate_2d = self.gate_net_2d(x_2d)  # [B*Z, 1, H, W]

        # [B*Z, 1, H, W] -> [B, 1, Z, H, W]
        gate_3d = gate_2d.view(B, 1, Z, H, W)

        return gate_3d

class DifferentialFeatureEnhancer(nn.Module):
    VALID_VARIANTS = ("full", "no_diff", "global_gate")

    def __init__(
        self,
        channels,
        reduction=8,
        z_kernel_size=3,
        variant="full",
    ):
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown DFE variant '{variant}'. "
                f"Expected one of {self.VALID_VARIANTS}."
            )
        self.kernel_size = (z_kernel_size, 1, 1)
        self.variant = variant
        
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta = nn.Parameter(torch.zeros(1))
        self.spatial_gate = SpatialGate(channels, reduction=8, kernel_size=3)

    def forward(self, features):
        """
        Args:
            features: [B,Z, C, H, W] (Detectron2 标准输出)
        Returns:
            out: [B,Z, C, H, W] (增强后的特征)
        """

        B,Z, C, H, W = features.shape
        x = features.permute(0, 2, 1, 3, 4) # [B, C, Z, H, W]

        # 2. 计算特征差分 (The High-Pass Filter)
        # Pad Z axis
        pad = self.kernel_size[0] // 2
        x_pad = F.pad(x, (0,0, 0,0, pad,pad), mode='replicate')
        
        # Smooth (Low-freq)
        x_smooth = F.avg_pool3d(x_pad, kernel_size=self.kernel_size, stride=1)
        
        # Diff (High-freq)
        x_diff = x - x_smooth
        
        gate_input = x if self.variant == "no_diff" else x_diff
        gate = self.spatial_gate(gate_input) # [B, 1, Z, H, W]
        if self.variant == "global_gate":
            # Retain plane-specific modulation while removing spatial adaptation.
            gate = gate.mean(dim=(-2, -1), keepdim=True)
        
        out = x * (self.alpha + self.beta * gate)
        
        # 5. 还原维度: [B, C, Z, H, W] -> [B, Z, C, H, W]
        out = out.permute(0, 2, 1, 3, 4)
        
        return out

class GaussianFocusModule(nn.Module):
    def __init__(self, hidden_dim=256, mlp_depth=3):
        """
        Args:
            hidden_dim: Transformer Query 的维度 (通常是 256)
            mlp_depth: MLP 的层数，3层通常效果最好
        """
        super().__init__()
        
        # 1. 定义 MLP 网络
        # 结构: Linear -> ReLU -> Linear -> ReLU -> Linear -> Output(2)
        layers = []
        for i in range(mlp_depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # 最后一层输出 2 个值: [mu_logit, sigma_logit]
        layers.append(nn.Linear(hidden_dim, 2))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化技巧: 最后一层 Bias 设为特定值有助于快速收敛
        # 让 mu 初始在中间，sigma 初始在 1.0 左右
        # (这步可选，PyTorch 默认初始化也能跑，但这样更稳)
        nn.init.constant_(self.mlp[-1].bias[0], 0.0) # Sigmoid(0) = 0.5 -> Z/2
        nn.init.constant_(self.mlp[-1].bias[1], 0.55) # Softplus(0.55) ≈ 1.0

    def forward(self, queries, current_z_len=None):
        """
        Args:
            queries: [B, Q, C] (Batch, Num_Queries, Channels)

        Returns:
            gaussian_weights: [B, Q, Z] (用于 Logit Bias)
            pred_mu: [B, Q] (用于 Loss)
            pred_sigma: [B, Q] (用于 Loss)
        """
        B, Q, C = queries.shape
        
        # 如果没有传入 Z 长度，给个默认值 (比如训练时的配置)
        Z = current_z_len
        
        # 1. MLP 预测
        params = self.mlp(queries) # [B, Q, 2]
        
        # 2. 激活函数与物理约束 (Crucial!)
        
        # --- Mu (位置) ---
        # Sigmoid 归一化到 [0, 1]，然后乘 Z-1 映射到物理坐标
        # 比如 Z=11, 范围就是 [0, 10]
        pred_mu_norm = torch.sigmoid(params[..., 0]) 
        pred_mu = pred_mu_norm * (Z - 1)
        
        # --- Sigma (厚度) ---
        # Softplus: log(1 + exp(x))，保证平滑且恒正
        # + 0.6: 硬下限保底 (和你 Mapper 里的逻辑对齐)
        pred_sigma = F.softplus(params[..., 1]) + 0.6
        
        # 3. 生成高斯分布权重
        # 构建 Z 轴网格: [0, 1, 2, ..., Z-1]
        # device 必须和 queries 一致
        z_grid = torch.arange(Z, device=queries.device, dtype=torch.float32)
        
        # 调整维度以便广播 (Broadcasting)
        # z_grid: [1, 1, Z]
        # mu, sigma: [B, Q, 1]
        z_grid = z_grid.view(1, 1, Z)
        mu = pred_mu.unsqueeze(-1)
        sigma = pred_sigma.unsqueeze(-1)
        
        # 高斯公式: exp( - (x - mu)^2 / 2sigma^2 )
        # 不包含前面的 1/(sigma*sqrt(2pi)) 系数，因为我们只关心相对权重做 Bias
        squared_diff = (z_grid - mu) ** 2
        div_term = 2 * (sigma ** 2) + 1e-6 # 防止除零
        
        gaussian_weights = torch.exp(-squared_diff / div_term)
        
        return gaussian_weights, pred_mu, pred_sigma

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class FocusMultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        # video related
        num_frames,
        use_gaussian_focus: bool=False,
        use_dfe: bool=False,
        use_focus_sampler: bool=False,
        dfe_variant: str="full",
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        self.num_frames = num_frames

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine3D(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.use_gaussian_focus = use_gaussian_focus
        self.use_dfe = use_dfe
        self.use_focus_sampler = use_focus_sampler
        self.dfe_variant = dfe_variant
        if self.use_focus_sampler:
            self.focus_sampler = FocusFeatureSampler(feature_channels=in_channels, hidden_dim=hidden_dim)
        if self.use_dfe:
            self.dfe = DifferentialFeatureEnhancer(
                channels=hidden_dim,
                variant=dfe_variant,
            )
        if self.use_gaussian_focus:
            self.gaussian_module = GaussianFocusModule(hidden_dim=hidden_dim)
    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        ret["num_frames"] = cfg.INPUT.SAMPLING_FRAME_NUM
        ret["use_gaussian_focus"] = cfg.MODEL.MASK_FORMER.USE_GAUSSIAN_FOCUS
        ret["use_dfe"] = cfg.MODEL.MASK_FORMER.USE_DFE
        ret["use_focus_sampler"] = cfg.MODEL.MASK_FORMER.USE_FOCUS_SAMPLER
        ret["dfe_variant"] = getattr(
            cfg.MODEL.MASK_FORMER,
            "DFE_VARIANT",
            "full",
        )
        return ret

    def forward(self, x, mask_features, mask = None, proposal_centers = None):
        bt, c_m, h_m, w_m = mask_features.shape
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs

        mask_features = mask_features.view(bs, t, c_m, h_m, w_m)
        if self.use_dfe:
            mask_features = self.dfe(mask_features)
        mask_features_for_sampler = mask_features.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
    
        if self.training and comm.is_main_process():
            storage = get_event_storage()
            if self.use_dfe:
                storage.put_scalar("dfe/alpha", self.dfe.alpha)
                storage.put_scalar("dfe/beta", self.dfe.beta)

        if self.use_focus_sampler:
            proposal_centers_tensor = proposal_centers.to(mask_features.device).view(bs, -1, 3)

            # 1. 提取坐标 Stack 成 Tensor [B, N, 3]
            proposal_centers = proposal_centers_tensor

            # 2. 从高分辨特征图采样 [B, N, C]
            sampled_query_content = self.focus_sampler(mask_features_for_sampler, proposal_centers)

            # 3. 混合初始化 (Hybrid): Sampled + Learnable Bias
            learnable_bias = self.query_feat.weight.unsqueeze(0).expand(bs, -1, -1)
            query_content = sampled_query_content + learnable_bias

            # Ensure shape matches [num_queries, batch, hidden_dim]
            initial_output = query_content.permute(1, 0, 2).contiguous()
        else:
            # Fallback: 如果没有 proposal_centers (比如推理时出错，或者旧代码逻辑)
            # 回退到原始 Mask2Former 逻辑
            initial_output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            feat = self.input_proj[i](x[i])
            pos_pe = self.pe_layer(
                x[i].view(bs, t, -1, size_list[-1][0], size_list[-1][1]),
                mask=None,
                # gamma=self.gammas[i % self.num_layers],
            ).flatten(3)
            src.append(feat.flatten(2) + self.level_embed.weight[i][None, :, None])
            pos.append(pos_pe)

            _, c, hw = src[-1].shape
            pos[-1] = pos[-1].view(bs, t, c, hw).permute(1, 3, 0, 2).flatten(0, 1)
            src[-1] = src[-1].view(bs, t, c, hw).permute(1, 3, 0, 2).flatten(0, 1)

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = initial_output

        predictions_class = []
        predictions_mask = []
        predictions_mu = []
        predictions_sigma = []
        predictions_g_weights = []
        # prediction heads on learnable query features
        if self.use_gaussian_focus:
            outputs_class, outputs_mask, attn_mask, pred_mu, pred_sigma,g_weights = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
            predictions_g_weights.append(g_weights)
            predictions_mu.append(pred_mu)
            predictions_sigma.append(pred_sigma)
        else:
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            if self.use_gaussian_focus:
                outputs_class, outputs_mask, attn_mask, pred_mu, pred_sigma,g_weights = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                predictions_mu.append(pred_mu)
                predictions_sigma.append(pred_sigma)
                predictions_g_weights.append(g_weights)
            else:
                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_mu': predictions_mu[-1] if self.use_gaussian_focus else None,
            'pred_sigma': predictions_sigma[-1] if self.use_gaussian_focus else None,
            'pred_g_weights': predictions_g_weights[-1] if self.use_gaussian_focus else None,
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, 
                predictions_mask,
                predictions_mu if self.use_gaussian_focus else None,
                predictions_sigma if self.use_gaussian_focus else None,
                predictions_g_weights if self.use_gaussian_focus else None,
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,btchw->bqthw", mask_embed, mask_features)
        b, q, t, _, _ = outputs_mask.shape
        if self.use_gaussian_focus:
            g_weights, pred_mu, pred_sigma = self.gaussian_module(decoder_output, current_z_len=t)
            g_weights = g_weights.detach()
            # g_bias = torch.log(g_weights + 1e-6)
        #     attn_mask = outputs_mask + g_bias.unsqueeze(-1).unsqueeze(-1)
        # else:
        #     attn_mask = outputs_mask

        # NOTE: prediction is of higher-resolution
        # [B, Q, T, H, W] -> [B, Q, T*H*W] -> [B, h, Q, T*H*W] -> [B*h, Q, T*HW]
        attn_mask = F.interpolate(outputs_mask.flatten(0, 1), size=attn_mask_target_size, mode="bilinear", align_corners=False).view(
            b, q, t, attn_mask_target_size[0], attn_mask_target_size[1])
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        # debug = {
        #     "logits": outputs_class.cpu(),
        #     "attn_mask": attn_mask.cpu(),
        #     # "pred_mu": pred_mu.detach().cpu(),
        #     # "pred_sigma": pred_sigma.detach().cpu(),
        # }
        # torch.save(debug, "debug.pt")
        
        if self.use_gaussian_focus:
            return outputs_class, outputs_mask, attn_mask, pred_mu, pred_sigma,g_weights
        else:
            return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_mu=None, outputs_sigma=None, outputs_g_weights=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            if outputs_mu is not None and outputs_sigma is not None:
                return [
                    {"pred_logits": a, "pred_masks": b, "pred_mu": c, "pred_sigma": d, "pred_g_weights": e}
                    for a, b, c, d, e in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_mu[:-1], outputs_sigma[:-1], outputs_g_weights[:-1])
                ]
            else:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
