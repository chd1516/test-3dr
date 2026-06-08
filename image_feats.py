import argparse
from tqdm import tqdm
import time
import random
import numpy as np

import torch
import clip
import os
from dataset.esb_core import ESBCoreDataset
from dataset.ntu_core import NTUCoreDataset
from dataset.mn40_core import MN40CoreDataset
from dataset.abo_core import ABOCoreDataset
from loralib.utils import (
    apply_lora,
    load_lora,
)
from train import fuse_image_point_features, unpack_multimodal_batch
from uni3d_branch import Uni3DPointEncoder


def setup_seed():
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"random seed: {seed}")


def extract_feats(model, data_loader, point_model=None, point_fusion_weight=0.5):
    """Extract CLIP multi-view features or fused CLIP+Uni3D descriptors."""
    model.eval()
    if point_model is not None:
        point_model.eval()
    feats = []
    labels = []
    for batch in tqdm(data_loader):
        mv_imgs, point_clouds, category, _ = unpack_multimodal_batch(batch)
        mv_imgs = mv_imgs.cuda()
        bz, n, c, h, w = mv_imgs.size()
        mv_imgs = mv_imgs.view(-1, c, h, w)
        mv_imgs = mv_imgs.half()

        mv_feat = model.encode_image(mv_imgs)
        mv_feat = mv_feat.view(bz, n, -1)

        if point_model is not None:
            if point_clouds is None:
                raise ValueError(
                    "Uni3D fusion is enabled, but the dataloader did not return point clouds. "
                    "Use --modality mv_point."
                )
            point_feat = point_model(point_clouds.cuda())
            mv_feat = fuse_image_point_features(
                mv_feat.mean(dim=1), point_feat, point_fusion_weight
            ).unsqueeze(1)

        feats.append(mv_feat.detach().cpu())
        labels.append(category.detach().cpu())

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


def build_point_model(args):
    if not args.use_uni3d:
        return None

    return Uni3DPointEncoder(
        checkpoint_path=args.uni3d_ckpt,
        pc_model=args.uni3d_pc_model,
        pretrained_pc=args.uni3d_pretrained_pc,
        pc_feat_dim=args.uni3d_pc_feat_dim,
        embed_dim=args.uni3d_embed_dim,
        group_size=args.uni3d_group_size,
        num_group=args.uni3d_num_group,
        pc_encoder_dim=args.uni3d_pc_encoder_dim,
        patch_dropout=args.uni3d_patch_dropout,
        drop_path_rate=args.uni3d_drop_path_rate,
        freeze=True,
    ).cuda()


def main():

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="esb",
        help="esb, ntu, mn40, abo",
    )
    parser.add_argument("--backbone", default="ViT-B/32", type=str)
    # Training arguments
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--n_iters", default=500, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    # LoRA arguments
    parser.add_argument(
        "--position",
        type=str,
        default="all",
        choices=["bottom", "mid", "up", "half-up", "half-bottom", "all", "top3"],
        help="where to put the LoRA modules",
    )
    parser.add_argument(
        "--encoder", type=str, choices=["text", "vision", "both"], default="both"
    )
    parser.add_argument(
        "--params",
        metavar="N",
        type=str,
        nargs="+",
        default=["q", "k", "v"],
        help="list of attention matrices where putting a LoRA",
    )
    parser.add_argument(
        "--r", default=2, type=int, help="the rank of the low-rank matrices"
    )
    parser.add_argument("--alpha", default=1, type=int, help="scaling (see LoRA paper)")
    parser.add_argument(
        "--dropout_rate",
        default=0.25,
        type=float,
        help="dropout rate applied before the LoRA module",
    )

    parser.add_argument(
        "--save_path",
        default="./",
        help="path to save the lora modules after training, not saved if None",
    )
    parser.add_argument(
        "--filename",
        default="lora_weights",
        help="file name to save the lora weights (.pt extension will be added)",
    )

    parser.add_argument(
        "--zero_shot",
        default=False,
    )

    parser.add_argument("--n_view", default=24, type=int)
    parser.add_argument(
        "--use_uni3d",
        default=False,
        action="store_true",
        help="Save fused CLIP+Uni3D descriptors instead of CLIP-only multi-view features.",
    )
    parser.add_argument(
        "--uni3d_ckpt", default=None, help="Optional Uni3D checkpoint path."
    )
    parser.add_argument("--uni3d_pc_model", default="eva02_base_patch14_448")
    parser.add_argument("--uni3d_pretrained_pc", default="")
    parser.add_argument("--uni3d_pc_feat_dim", default=768, type=int)
    parser.add_argument("--uni3d_embed_dim", default=512, type=int)
    parser.add_argument("--uni3d_group_size", default=32, type=int)
    parser.add_argument("--uni3d_num_group", default=512, type=int)
    parser.add_argument("--uni3d_pc_encoder_dim", default=256, type=int)
    parser.add_argument("--uni3d_patch_dropout", default=0.0, type=float)
    parser.add_argument("--uni3d_drop_path_rate", default=0.0, type=float)
    parser.add_argument("--point_fusion_weight", default=0.5, type=float)

    args = parser.parse_args()
    print(args)

    setup_seed()

    if args.dataset == "esb":
        data_dir = "/data/cd/data/3dor/OS-ESB-core"
        train_dataset = ESBCoreDataset(
            data_dir,
            "train",
            modality="mv_point" if args.use_uni3d else "mv",
            n_view=args.n_view,
        )
        query_dataset = ESBCoreDataset(
            data_dir,
            "query",
            modality="mv_point" if args.use_uni3d else "mv",
            n_view=args.n_view,
        )
        target_dataset = ESBCoreDataset(
            data_dir,
            "target",
            modality="mv_point" if args.use_uni3d else "mv",
            n_view=args.n_view,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=0
        )

    elif args.dataset == "ntu":
        data_dir = "/data/cd/data/3dor/OS-NTU-core"
        train_dataset = NTUCoreDataset(
            data_dir,
            "train",
            modality="mv_point" if args.use_uni3d else "mv",
            n_view=args.n_view,
        )
        query_dataset = NTUCoreDataset(
            data_dir,
            "query",
            modality="mv_point" if args.use_uni3d else "mv",
            n_view=args.n_view,
        )
        target_dataset = NTUCoreDataset(
            data_dir,
            "target",
            modality="mv_point" if args.use_uni3d else "mv",
            n_view=args.n_view,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=0
        )

    elif args.dataset == "mn40":
        data_dir = "/data/cd/data/3dor/OS-MN40-core"
        train_dataset = MN40CoreDataset(
            data_dir,
            "train",
            modality="mv_point" if args.use_uni3d else "mv",
            n_view=args.n_view,
        )
        query_dataset = MN40CoreDataset(
            data_dir,
            "query",
            modality="mv_point" if args.use_uni3d else "mv",
            n_view=args.n_view,
        )
        target_dataset = MN40CoreDataset(
            data_dir,
            "target",
            modality="mv_point" if args.use_uni3d else "mv",
            n_view=args.n_view,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=False, num_workers=0
        )

    elif args.dataset == "abo":
        data_dir = "/data/cd/data/3dor/OS-ABO-core"
        train_dataset = ABOCoreDataset(
            data_dir,
            "train",
            modality="mv_point" if args.use_uni3d else "mv",
            n_view=args.n_view,
        )
        query_dataset = ABOCoreDataset(
            data_dir,
            "query",
            modality="mv_point" if args.use_uni3d else "mv",
            n_view=args.n_view,
        )
        target_dataset = ABOCoreDataset(
            data_dir,
            "target",
            modality="mv_point" if args.use_uni3d else "mv",
            n_view=args.n_view,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=0
        )
    else:
        raise NotImplementedError
    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    model_clip, _ = clip.load(args.backbone)
    model_clip.eval()
    if args.zero_shot == False:
        list_lora_layers = apply_lora(args, model_clip)
        model_clip = model_clip.cuda()
        load_lora(args, list_lora_layers)
    else:
        model_clip = model_clip.cuda()

    point_model = build_point_model(args)

    if True:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            # save features
            save_file_prefix = f"{args.dataset}"
            save_file_suffix = f"{args.backbone}".replace("/", "_")
            save_file_suffix += f"_{args.r}"
            if args.zero_shot:
                save_file_suffix += "_zs"
            if args.use_uni3d:
                save_file_suffix += "_uni3d"
            feats_query, labels_query = extract_feats(
                model_clip, query_loader, point_model, args.point_fusion_weight
            )
            os.makedirs("output/image_feats", exist_ok=True)
            np.save(
                f"output/image_feats/{args.dataset}_query_feats_{save_file_suffix}.npy",
                feats_query.numpy(),
            )
            np.save(
                f"output/image_feats/{args.dataset}_query_labels_{save_file_suffix}.npy",
                labels_query.numpy(),
            )

            feats_target, labels_target = extract_feats(
                model_clip, target_loader, point_model, args.point_fusion_weight
            )
            np.save(
                f"output/image_feats/{args.dataset}_target_feats_{save_file_suffix}.npy",
                feats_target.numpy(),
            )
            np.save(
                f"output/image_feats/{args.dataset}_target_labels_{save_file_suffix}.npy",
                labels_target.numpy(),
            )
            print("Extract features done!")


if __name__ == "__main__":

    all_st = time.time()
    main()
    all_sec = time.time() - all_st
    print(
        f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes {all_sec%60:.2f}s!"
    )
    print("All done!")
