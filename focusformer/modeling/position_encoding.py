# Copyright (c) Facebook, Inc. and its affiliates.
# # Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
"""
Various positional encodings for the transformer.
"""
import math
from typing import Optional

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        # b, t, c, h, w
        assert x.dim() == 5, f"{x.shape} should be a 5-dimensional Tensor, got {x.dim()}-dimensional Tensor instead"
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(1), x.size(3), x.size(4)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        dim_t_z = torch.arange((self.num_pos_feats * 2), dtype=torch.float32, device=x.device)
        dim_t_z = self.temperature ** (2 * (dim_t_z // 2) / (self.num_pos_feats * 2))

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t_z
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos = (torch.cat((pos_y, pos_x), dim=4) + pos_z).permute(0, 1, 4, 2, 3)  # b, t, c, h, w
        return pos


class PerQueryShiftPE(nn.Module):
    def __init__(self, num_frames, hidden_dim, max_shift_ratio=0.25):
        super().__init__()
        self.num_frames = num_frames
        self.max_shift = num_frames * max_shift_ratio
        self.delta_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, query_embed, z_vals, z_center, pe_z_fn):
        """
        query_embed: [Nq, D]
        z_vals: [T] or [1,T]
        pe_z_fn: function that maps (|z - z_center|) -> PE_z vector
        """
        delta_z = torch.tanh(self.delta_head(query_embed)) * self.max_shift  # [Nq,1]
        z_center_q = z_center + delta_z  # per-query center
        dist = torch.abs(z_vals.unsqueeze(0) - z_center_q)  # [Nq,T]
        pe_z_q = pe_z_fn(dist.unsqueeze(-1))  # [Nq,T,Cz]
        return pe_z_q

class AsymmetricFocalPositionalEncoding(nn.Module):
    """
    Asymmetric Focus Positional Encoding (AFPE) for multi-focal Z-stack microscopy images.
    
    This positional encoding combines:
    1. Standard sinusoidal positional encoding for X, Y spatial dimensions
    2. Asymmetric focus positional encoding for Z-axis with learnable asymmetry
    
    Args:
        num_pos_feats (int): Number of positional features for X, Y dimensions. Default: 64.
        temperature (int): Temperature parameter for sinusoidal encoding. Default: 10000.
        normalize (bool): Whether to normalize positional encodings. Default: False.
        scale (Optional[float]): Scale factor for normalization. Default: None.
        num_embeddings (int): Maximum number of Z-stack layers (D).
        z_center (Optional[float]): The theoretical central focal plane index. 
                                   If None, defaults to (num_embeddings-1)/2.0.
        gamma_init (float): Initial value for the learnable asymmetry parameter gamma. Default: 0.0.
        hidden_mlp_dim (int): Hidden dimension for the Z-axis MLP. Default: num_pos_feats.
    
    Example:
        >>> afpe_encoder = AsymmetricFocalPositionalEncoding(
        ...     num_pos_feats=64,
        ...     num_embeddings=100, 
        ...     z_center=49.5
        ... )
        >>> x = torch.randn(2, 100, 256, 32, 32)  # b, t, c, h, w
        >>> pos_encoding = afpe_encoder(x)  # Shape: [2, 100, 192, 32, 32] (64*2 + 64 for Z)
    """
    
    def __init__(
        self,
        num_pos_feats=64,
        temperature=10000,
        normalize=True,
        scale = None,
        num_embeddings=100,
        hidden_mlp_dim=None,
        z_center=None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        
        # Z-axis specific parameters
        self.num_embeddings = num_embeddings
        if z_center is None:
            z_center = (num_embeddings - 1) / 2.0
        self.z_center = z_center
        # MLP for Z-axis positional encoding
        if hidden_mlp_dim is None:
            hidden_mlp_dim = num_pos_feats

        # Z-axis MLP + normalization for asymmetric focal encoding
        self.mlp_z = nn.Sequential(
            nn.Linear(1, hidden_mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp_dim, num_pos_feats * 2),
            # nn.LayerNorm(num_pos_feats * 2)
        )
        # learnable scale to balance XY and Z positional magnitudes
        self.scale_z = nn.Parameter(torch.ones(1, num_pos_feats * 2)*2.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        gamma: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the Asymmetric Focus Positional Encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape [b, t, c, h, w] where t is the Z-stack dimension.
            mask (Optional[torch.Tensor]): Optional mask tensor of shape [b, t, h, w].
        
        Returns:
            torch.Tensor: Positional encoding tensor of shape [b, t, c, h, w] where c = num_pos_feats * 3.
        """
        # b, t, c, h, w
        assert x.dim() == 5, f"{x.shape} should be a 5-dimensional Tensor, got {x.dim()}-dimensional Tensor instead"
        B, T, _, H, W = x.shape
        if mask is None:
            mask = torch.zeros((B, T, H, W), device=x.device, dtype=torch.bool)
        not_mask = ~mask

        # === (1) XY sinusoidal encoding ===
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=5).flatten(4)
        pos_xy = torch.cat((pos_y, pos_x), dim=4).permute(0, 1, 4, 2, 3)  # [B, T, 2C, H, W]

        # === (2) Z-axis asymmetric encoding ===
        z_vals = torch.arange(T, dtype=torch.float32, device=x.device).view(1, T, 1)
        gamma = 0.0 if gamma is None else gamma

        dist = torch.abs(z_vals - self.z_center).unsqueeze(1) + (gamma * (z_vals - self.z_center)).unsqueeze(1)
        dist = dist.view(1, T, 1, 1, 1).expand(B, -1, -1, H, W)
        dist = dist.permute(0, 1, 3, 4, 2).contiguous()  # [B, T, H, W, 1]

        # z_pe = self.mlp_z(dist)  # [B, T, H, W, 2C]
        # z_pe = z_pe * self.scale_z.view(1, 1, 1, 1, -1)
        z_pe = torch.tanh(self.mlp_z(dist)) * self.scale_z.view(1,1,1,1,-1)
        # z_pe = z_pe + 0.1 * torch.tanh(dist.unsqueeze(-1))

        z_pe = z_pe.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, 2C, H, W]

        # 轻微残差项提供非对称形状
        # === (3) Combine ===
        pos = pos_xy + z_pe
        return pos