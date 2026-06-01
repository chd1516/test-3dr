"""Utilities for plugging the official Uni3D point-cloud encoder into 3DOR.

The official Uni3D repository is intentionally loaded lazily from a user-provided
path so this project can keep running its existing CLIP-only code when Uni3D's
extra dependencies (for example pointnet2_ops) are not installed.
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class Uni3DPointEncoder(nn.Module):
    """Thin adapter around Uni3D's official ``create_uni3d`` entry point.

    Uni3D expects point clouds as ``[B, N, 6]`` with XYZ and color channels.
    The current 3DOR datasets provide normalized XYZ only, so this adapter
    appends zero RGB channels before calling ``encode_pc``.
    """

    def __init__(
        self,
        repo_path,
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
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.checkpoint_path = checkpoint_path
        self.freeze = freeze

        self.model = self._build_official_model(
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

    def _build_official_model(self, **kwargs):
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"Uni3D repository path does not exist: {self.repo_path}. "
                "Clone https://github.com/baaivision/Uni3D and pass "
                "--uni3d_repo_path /path/to/Uni3D."
            )

        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            uni3d_module = importlib.import_module("models.uni3d")
        except Exception as exc:
            raise ImportError(
                "Failed to import Uni3D's official models.uni3d module. "
                "Please install Uni3D requirements, including timm and "
                "pointnet2_ops, then pass --uni3d_repo_path correctly."
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
