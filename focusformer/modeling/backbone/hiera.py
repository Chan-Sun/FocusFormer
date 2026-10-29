"""Detectron2 wrapper for the native SAM2.1 Hiera-S image encoder."""

import logging
from pathlib import Path
from typing import Dict

import torch

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from .sam2_hiera import FpnNeck, Hiera, ImageEncoder, PositionEmbeddingSine


logger = logging.getLogger(__name__)


@BACKBONE_REGISTRY.register()
class SAM2HieraSBackbone(Backbone):
    """SAM2.1 Hiera-S with its native FPN neck exposed as res2--res5."""

    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()
        del input_shape

        trunk = Hiera(
            embed_dim=96,
            num_heads=1,
            stages=(1, 2, 11, 2),
            global_att_blocks=(7, 10, 13),
            window_pos_embed_bkg_spatial_size=(7, 7),
        )
        neck = FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                scale=None,
                temperature=10000,
            ),
            d_model=256,
            backbone_channel_list=[768, 384, 192, 96],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        )
        # SAM2 normally discards its lowest-resolution level (scalp=1).
        # FocusFormer consumes four levels, so retain the full native pyramid.
        self.image_encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=0)

        self._out_features = list(cfg.MODEL.HIERA.OUT_FEATURES)
        self._out_feature_channels = {name: 256 for name in self._out_features}
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._size_divisibility = 32

        if getattr(cfg.MODEL.HIERA, "LOAD_PRETRAINED", True):
            self._load_sam2_image_encoder(cfg.MODEL.HIERA.CHECKPOINT)

    def _load_sam2_image_encoder(self, checkpoint_path: str) -> None:
        checkpoint_file = Path(checkpoint_path).expanduser()
        if not checkpoint_file.is_file():
            raise FileNotFoundError(
                f"SAM2.1 Hiera-S checkpoint not found: {checkpoint_file}"
            )

        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("model", checkpoint)
        prefix = "image_encoder."
        image_encoder_state = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not image_encoder_state:
            raise KeyError(
                f"No '{prefix}' weights found in checkpoint: {checkpoint_file}"
            )

        incompatible = self.image_encoder.load_state_dict(
            image_encoder_state,
            strict=False,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "SAM2 image encoder checkpoint mismatch. "
                f"Missing: {incompatible.missing_keys}; "
                f"unexpected: {incompatible.unexpected_keys}"
            )
        logger.info(
            "Loaded %d SAM2.1 Hiera-S image encoder tensors from %s",
            len(image_encoder_state),
            checkpoint_file,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] not in (1, 3):
            raise ValueError(
                "SAM2HieraSBackbone expects NCHW input with one or three channels, "
                f"but received shape {tuple(x.shape)}"
            )
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.image_encoder(x)["backbone_fpn"]
        if len(features) != 4:
            raise RuntimeError(f"Expected four Hiera levels, got {len(features)}")
        return dict(zip(("res2", "res3", "res4", "res5"), features))

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility
