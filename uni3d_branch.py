"""Local Uni3D point-cloud branch used by the 3DOR training pipeline.

The repository already vendors the Uni3D model code in ``3d_model/``.  This
adapter keeps that implementation behind a small, CLIP-compatible interface so
existing multi-view training can opt into a native 3D branch without depending on
an external checkout.
"""

from pathlib import Path
from types import SimpleNamespace
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class Uni3DPointEncoder(nn.Module):
    """Thin adapter around the vendored Uni3D ``create_uni3d`` entry point.

    Uni3D expects point clouds as ``[B, N, 6]`` with XYZ and color channels.
    The current 3DOR datasets provide normalized XYZ only, so this adapter
    appends zero RGB channels before calling ``encode_pc``.  Returned features
    are L2-normalized to match CLIP image/text embeddings.
    """

    def __init__(
        self,
        checkpoint_path=None,
        pc_model="eva02_base_patch14_448",
        pretrained_pc="",
        pc_feat_dim=768,
        embed_dim=512,
        group_size=32,
        num_group=512,
        pc_encoder_dim=256,
        patch_dropout=0.0,
        drop_path_rate=0.0,
        freeze=True,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.freeze = freeze

        self.model = self._build_local_model(
            pc_model=pc_model,
            pretrained_pc=pretrained_pc,
            pc_feat_dim=pc_feat_dim,
            embed_dim=embed_dim,
            group_size=group_size,
            num_group=num_group,
            pc_encoder_dim=pc_encoder_dim,
            patch_dropout=patch_dropout,
            drop_path_rate=drop_path_rate,
        )
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def _build_local_model(self, **kwargs):
        model_dir = Path(__file__).resolve().parent / "3d_model"
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Vendored Uni3D model folder not found: {model_dir}"
            )

        try:
            uni3d_module = importlib.import_module("3d_model.uni3d")
        except Exception as exc:
            raise ImportError(
                "Failed to import vendored Uni3D code from 3d_model/. Install "
                "the point-cloud dependencies (notably timm and pointnet2_ops) "
                "before enabling --use_uni3d."
            ) from exc

        args = SimpleNamespace(**kwargs)
        return uni3d_module.create_uni3d(args)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint
        for key in ("state_dict", "model", "module"):
            if isinstance(checkpoint, dict) and key in checkpoint:
                state_dict = checkpoint[key]
                break

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key
            for prefix in ("module.", "model."):
                if cleaned_key.startswith(prefix):
                    cleaned_key = cleaned_key[len(prefix) :]
            cleaned_state_dict[cleaned_key] = value

        missing, unexpected = self.model.load_state_dict(
            cleaned_state_dict, strict=False
        )
        if missing:
            print(f"[Uni3D] Missing checkpoint keys: {len(missing)}")
        if unexpected:
            print(f"[Uni3D] Unexpected checkpoint keys: {len(unexpected)}")

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, point_clouds):
        if point_clouds is None:
            raise ValueError("Point clouds are required when Uni3D is enabled.")
        if point_clouds.size(-1) == 3:
            colors = torch.zeros_like(point_clouds)
            point_clouds = torch.cat([point_clouds, colors], dim=-1)
        elif point_clouds.size(-1) != 6:
            raise ValueError(
                "Uni3D expects point clouds with 3 or 6 channels, "
                f"got {point_clouds.size(-1)}."
            )

        features = self.model.encode_pc(point_clouds.float())
        return F.normalize(features, dim=-1)
