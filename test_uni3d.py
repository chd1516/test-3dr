"""Smoke-test the local Uni3D point-cloud branch.

This script builds the vendored Uni3D encoder through ``Uni3DPointEncoder`` and
runs a synthetic point-cloud forward pass. It is intentionally lightweight by
default (tiny ViT backbone and small point groups) so it can be used as a quick
sanity check before wiring Uni3D into the full 3DOR training/evaluation flow.

Example:
    python test_uni3d.py --device cuda

For a checkpoint/backbone that matches your experiment, override ``--pc_model``
and ``--uni3d_ckpt``.
"""

import argparse
import random

import torch

from uni3d_branch import Uni3DPointEncoder


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )
    return device


def make_synthetic_point_cloud(batch_size, num_points, with_color, device):
    xyz = torch.randn(batch_size, num_points, 3, device=device)
    xyz = xyz - xyz.mean(dim=1, keepdim=True)
    radius = xyz.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-6)
    xyz = xyz / radius.unsqueeze(-1)

    if not with_color:
        return xyz

    rgb = torch.rand(batch_size, num_points, 3, device=device)
    return torch.cat([xyz, rgb], dim=-1)


def assert_valid_embedding(features, batch_size, embed_dim, atol):
    expected_shape = (batch_size, embed_dim)
    if tuple(features.shape) != expected_shape:
        raise AssertionError(
            f"Unexpected Uni3D output shape {tuple(features.shape)}; "
            f"expected {expected_shape}."
        )
    if not torch.isfinite(features).all():
        raise AssertionError("Uni3D output contains NaN or Inf values.")

    norms = features.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=atol):
        raise AssertionError(
            "Uni3DPointEncoder should return L2-normalized features; "
            f"got norms {norms.detach().cpu().tolist()}."
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Smoke-test the vendored Uni3D branch."
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_points", default=128, type=int)
    parser.add_argument("--num_group", default=8, type=int)
    parser.add_argument("--group_size", default=8, type=int)
    parser.add_argument(
        "--pc_model",
        default="vit_tiny_patch16_224",
        help="timm ViT backbone used inside Uni3D for this smoke test.",
    )
    parser.add_argument(
        "--pc_feat_dim",
        default=192,
        type=int,
        help="Transformer width for --pc_model; vit_tiny_patch16_224 uses 192.",
    )
    parser.add_argument("--embed_dim", default=512, type=int)
    parser.add_argument("--pc_encoder_dim", default=128, type=int)
    parser.add_argument("--patch_dropout", default=0.0, type=float)
    parser.add_argument("--drop_path_rate", default=0.0, type=float)
    parser.add_argument(
        "--pretrained_pc",
        default="",
        help="Optional timm checkpoint_path for the point transformer backbone.",
    )
    parser.add_argument(
        "--uni3d_ckpt",
        default=None,
        help="Optional Uni3D checkpoint loaded into the wrapped Uni3D model.",
    )
    parser.add_argument(
        "--with_color",
        action="store_true",
        help="Feed synthetic XYZRGB point clouds. By default the test feeds XYZ only.",
    )
    parser.add_argument(
        "--norm_atol",
        default=1e-4,
        type=float,
        help="Tolerance for checking L2-normalized output embeddings.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.num_points < args.group_size or args.num_points < args.num_group:
        raise ValueError("--num_points must be >= both --group_size and --num_group.")

    setup_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[Uni3D smoke] device={device}")
    print(
        "[Uni3D smoke] config="
        f"pc_model={args.pc_model}, pc_feat_dim={args.pc_feat_dim}, "
        f"embed_dim={args.embed_dim}, num_group={args.num_group}, "
        f"group_size={args.group_size}, num_points={args.num_points}"
    )

    model = Uni3DPointEncoder(
        checkpoint_path=args.uni3d_ckpt,
        pc_model=args.pc_model,
        pretrained_pc=args.pretrained_pc,
        pc_feat_dim=args.pc_feat_dim,
        embed_dim=args.embed_dim,
        group_size=args.group_size,
        num_group=args.num_group,
        pc_encoder_dim=args.pc_encoder_dim,
        patch_dropout=args.patch_dropout,
        drop_path_rate=args.drop_path_rate,
        freeze=True,
    ).to(device)
    model.eval()

    point_cloud = make_synthetic_point_cloud(
        args.batch_size, args.num_points, args.with_color, device
    )
    print(f"[Uni3D smoke] input_shape={tuple(point_cloud.shape)}")

    with torch.no_grad():
        features = model(point_cloud)

    assert_valid_embedding(features, args.batch_size, args.embed_dim, args.norm_atol)
    print(f"[Uni3D smoke] output_shape={tuple(features.shape)}")
    print(
        "[Uni3D smoke] output_norms="
        f"{[round(v, 6) for v in features.norm(dim=-1).detach().cpu().tolist()]}"
    )
    print("[Uni3D smoke] PASS: Uni3D model can be built and run a forward pass.")


if __name__ == "__main__":
    main()
