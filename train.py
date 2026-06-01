import sys
import argparse
from tqdm import tqdm
import time
import random
import numpy as np
import scipy.spatial
from misc_utils.metric_tools import acc_score, map_score, eval_all_metric
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(".")
from dataset.feat_dataset import FeatDataset
import clip
from dataset.esb_core import ESBCoreDataset
from dataset.ntu_core import NTUCoreDataset
from dataset.mn40_core import MN40CoreDataset
from dataset.abo_core import ABOCoreDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import clip
from loralib.utils import (
    mark_only_lora_as_trainable,
    apply_lora,
    get_lora_parameters,
    lora_state_dict,
    save_lora,
    load_lora,
)
from loralib.layers import PlainMultiheadAttentionLoRA
from uni3d_branch import Uni3DPointEncoder


def setup_seed():
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print(f"random seed: {seed}")


def unpack_multimodal_batch(batch):
    """Return multi-view images, optional point clouds, labels, and paths.

    Datasets can run in legacy ``mv`` mode (images, label, path) or in
    ``mv_point`` mode (images, point_cloud, label, path). Keeping this helper
    central lets the current CLIP-only training path ignore point clouds while
    the upcoming 3D branch can consume the same dataloader output.
    """
    if len(batch) == 4:
        mv_imgs, point_clouds, category, paths = batch
        return mv_imgs, point_clouds, category, paths

    mv_imgs, category, paths = batch
    return mv_imgs, None, category, paths


def extract_feats(model, data_loader, point_model=None, point_weight=0.5):
    model.eval()
    feats = []
    labels = []
    for batch in tqdm(data_loader):
        mv_imgs, _, category, _ = unpack_multimodal_batch(batch)
        mv_imgs = mv_imgs.cuda()
        bz, n, c, h, w = mv_imgs.size()
        mv_imgs = mv_imgs.view(-1, c, h, w)
        mv_imgs = mv_imgs.half()
        mv_feat = model.encode_image(mv_imgs)
        mv_feat = F.normalize(mv_feat, dim=-1)
        mv_feat = mv_feat.view(bz, n, -1)

        if point_model is not None:
            if point_model.training:
                point_model.eval()
            _, point_clouds, _, _ = unpack_multimodal_batch(batch)
            point_clouds = point_clouds.cuda()
            pc_feat = point_model(point_clouds)
            mv_feat = fuse_image_point_features(
                mv_feat.mean(dim=1), pc_feat, point_weight
            ).unsqueeze(1)

        feats.append(mv_feat.detach().cpu())
        labels.append(category.detach().cpu())

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


@torch.no_grad()
def test_model_clip_osr3d_feats(
    query_loader,
    target_loader,
    model,
    is_clip=True,
    eval_all=False,
):
    # set model to eval mode
    model.eval()

    query_feats = []
    query_labels = []
    for mv_feats, category in tqdm(query_loader):
        mv_feats = mv_feats.cuda()

        if is_clip:
            mv_feats_mean = mv_feats.mean(dim=1)
            feats = mv_feats_mean

        query_feats.append(feats.detach().cpu())
        query_labels.append(category.detach().cpu())

    query_feats = torch.cat(query_feats, dim=0).cuda()
    query_labels = torch.cat(query_labels, dim=0).cuda()

    # target
    target_feats = []
    target_labels = []
    for mv_feats, category in tqdm(target_loader):
        mv_feats = mv_feats.cuda()

        if is_clip:
            mv_feats_mean = mv_feats.mean(dim=1)
            feats = mv_feats_mean

        target_feats.append(feats.detach().cpu())
        target_labels.append(category.detach().cpu())

    target_feats = torch.cat(target_feats, dim=0).cuda()
    target_labels = torch.cat(target_labels, dim=0).cuda()

    # retrieval evaluation
    v, v = retrieval_eval(
        query_feats, target_feats, query_labels, target_labels, eval_all
    )
    print("test_model_clip_osr3d passed")
    return v


@torch.no_grad()
def retrieval_eval(query, target, query_lbls, target_lbls, eval_all=False):
    query_fts = query.squeeze().detach().cpu().numpy()
    target_fts = target.squeeze().detach().cpu().numpy()
    query_lbls = query_lbls.detach().cpu().numpy()
    target_lbls = target_lbls.detach().cpu().numpy()
    dist_mat = scipy.spatial.distance.cdist(query_fts, target_fts, "cosine")
    map_s = map_score(dist_mat, query_lbls, target_lbls)
    print(f"\t -> mAP: {map_s:.5f}")
    # evaluate all metrics
    if eval_all:
        print("evaluate all metrics:")
        eval_all_metric(query_fts, target_fts, query_lbls, target_lbls)
    return map_s, map_s


def build_uni3d_branch(args):
    if not args.use_uni3d:
        return None
    if args.modality != "mv_point":
        raise ValueError(
            "--use_uni3d requires --modality mv_point so point clouds are available."
        )
    if not args.uni3d_repo_path:
        raise ValueError(
            "--use_uni3d requires --uni3d_repo_path pointing to a local clone of "
            "https://github.com/baaivision/Uni3D."
        )

    uni3d_model = Uni3DPointEncoder(
        repo_path=args.uni3d_repo_path,
        checkpoint_path=args.uni3d_checkpoint,
        pc_model=args.uni3d_pc_model,
        pretrained_pc=args.uni3d_pretrained_pc,
        pc_feat_dim=args.uni3d_pc_feat_dim,
        embed_dim=args.uni3d_embed_dim,
        group_size=args.uni3d_group_size,
        num_group=args.uni3d_num_group,
        pc_encoder_dim=args.uni3d_pc_encoder_dim,
        patch_dropout=args.uni3d_patch_dropout,
        drop_path_rate=args.uni3d_drop_path_rate,
        freeze=args.freeze_uni3d,
    )
    return uni3d_model.cuda()


def fuse_image_point_features(image_features, point_features, point_weight):
    if point_features is None:
        return image_features
    fused_features = image_features + point_weight * point_features
    return F.normalize(fused_features, dim=-1)


def run_lora(
    args,
    clip_model,
    logit_scale,
    seen_classnames,
    train_loader,
    query_loader,
    target_loader,
    point_model=None,
):
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()

    mark_only_lora_as_trainable(clip_model)
    trainable_parameters = list(get_lora_parameters(clip_model))
    if point_model is not None and not args.freeze_uni3d:
        trainable_parameters.extend(
            param for param in point_model.parameters() if param.requires_grad
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        lr=args.lr,
    )
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[100, 300], gamma=0.1
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epoch, eta_min=1e-6
    )

    scaler = torch.amp.GradScaler("cuda")
    # 初始化保存最高mAp的变量
    best_mAp = 0.0

    for epoch in range(args.epoch):
        clip_model.train()
        loss_epoch = 0.0

        for batch in tqdm(train_loader):
            images, point_clouds, target, _ = unpack_multimodal_batch(batch)
            images, target = images.cuda(), target.cuda()
            if point_clouds is not None:
                point_clouds = point_clouds.cuda()

            template = "A synthetic 3D model view of {} with different angle."
            texts = [
                template.format(classname.replace("_", " "))
                for classname in seen_classnames
            ]

            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                texts = clip.tokenize(texts).cuda()
                class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                bz, n, c, h, w = images.size()
                images = images.view(-1, c, h, w)
                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                image_features = image_features.view(bz, n, -1)
                image_features = image_features.mean(dim=1)
                image_features = F.normalize(image_features, dim=-1)
                cosine_similarity = logit_scale * image_features @ text_features.t()
                loss = F.cross_entropy(cosine_similarity, target)

                if point_model is not None:
                    point_features = point_model(point_clouds)
                    point_similarity = logit_scale * point_features @ text_features.t()
                    loss = loss + args.uni3d_loss_weight * F.cross_entropy(
                        point_similarity, target
                    )
                    fused_features = fuse_image_point_features(
                        image_features, point_features, args.uni3d_fusion_weight
                    )
                    fused_similarity = logit_scale * fused_features @ text_features.t()
                    loss = loss + args.uni3d_fused_loss_weight * F.cross_entropy(
                        fused_similarity, target
                    )

            loss_epoch += loss.item() * target.shape[0]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        print(f"Epoch {epoch + 1}: Loss = {loss_epoch / len(train_loader.dataset)}")

        clip_model.eval()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            feats_query, labels_query = extract_feats(
                clip_model, query_loader, point_model, args.uni3d_fusion_weight
            )
            feats_target, labels_target = extract_feats(
                clip_model, target_loader, point_model, args.uni3d_fusion_weight
            )
            query_dataset = FeatDataset(feats_query, labels_query)
            target_dataset = FeatDataset(feats_target, labels_target)
            query_loader_epoch = torch.utils.data.DataLoader(
                query_dataset, batch_size=16, shuffle=False, num_workers=0
            )
            target_loader_epoch = torch.utils.data.DataLoader(
                target_dataset, batch_size=16, shuffle=False, num_workers=0
            )
            mAp = test_model_clip_osr3d_feats(
                query_loader_epoch, target_loader_epoch, clip_model, is_clip=True
            )
            if mAp > best_mAp:
                best_mAp = mAp
                save_lora(args, list_lora_layers)
            print(f"Best mAp: {best_mAp}")


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
    parser.add_argument("--epoch", default=10, type=int)
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
        "--r", default=4, type=int, help="the rank of the low-rank matrices"
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
        "--eval_only",
        default=False,
        action="store_true",
        help="only evaluate the LoRA modules (save_path should not be None)",
    )

    parser.add_argument("--n_view", default=24, type=int)
    parser.add_argument(
        "--modality",
        default="mv_point",
        choices=["mv", "mv_point"],
        help="Input modality for dataloaders: legacy multi-view only or multi-view plus point cloud.",
    )
    parser.add_argument(
        "--use_uni3d", action="store_true", help="Enable the Uni3D point-cloud branch."
    )
    parser.add_argument(
        "--uni3d_repo_path",
        default="",
        help="Path to a local clone of https://github.com/baaivision/Uni3D.",
    )
    parser.add_argument(
        "--uni3d_checkpoint", default=None, help="Optional Uni3D checkpoint path."
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
    parser.add_argument("--uni3d_loss_weight", default=1.0, type=float)
    parser.add_argument("--uni3d_fused_loss_weight", default=0.5, type=float)
    parser.add_argument("--uni3d_fusion_weight", default=0.5, type=float)
    parser.add_argument(
        "--freeze_uni3d", default=True, action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()
    print(args)

    setup_seed()

    clip_model, _ = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100

    # create dataset

    # You can replace seen_classnames with descriptions or attributes generated by a large language model

    if args.dataset == "esb":
        seen_classnames = [
            "flat-thin wallcomponents___clips",
            "flat-thin wallcomponents___contact switches",
            "flat-thin wallcomponents___miscellaneous",
            "flat-thin wallcomponents___slender thin plates",
            "rectangular-cubic prism___bearing blocks",
            "rectangular-cubic prism___contoured surfaces",
            "rectangular-cubic prism___l blocks",
            "rectangular-cubic prism___machined blocks",
            "rectangular-cubic prism___motor bodies",
            "rectangular-cubic prism___rocker arms",
            "solid of revolution___90 degree elbows",
            "solid of revolution___container like parts",
            "solid of revolution___intersecting pipes",
            "solid of revolution___non-90 degree elbows",
            "solid of revolution___oil pans",
            "solid of revolution___posts",
            "solid of revolution___spoked wheels",
        ]

        data_dir = "/data/cd/data/3dor/OS-ESB-core"
        train_dataset = ESBCoreDataset(
            data_dir, "train", modality=args.modality, n_view=args.n_view
        )
        query_dataset = ESBCoreDataset(
            data_dir, "query", modality=args.modality, n_view=args.n_view
        )
        target_dataset = ESBCoreDataset(
            data_dir, "target", modality=args.modality, n_view=args.n_view
        )

    elif args.dataset == "ntu":
        seen_classnames = [
            "ball",
            "balloon",
            "book",
            "cannon",
            "cold_weapon___stick",
            "frame",
            "gun___pistol",
            "headstone",
            "plane___delta_wing",
            "plant___leaf",
            "plant___with_pot",
            "table___square",
            "watch",
        ]

        data_dir = "/data/cd/data/3dor/OS-NTU-core"
        train_dataset = NTUCoreDataset(
            data_dir, "train", modality=args.modality, n_view=args.n_view
        )
        query_dataset = NTUCoreDataset(
            data_dir, "query", modality=args.modality, n_view=args.n_view
        )
        target_dataset = NTUCoreDataset(
            data_dir, "target", modality=args.modality, n_view=args.n_view
        )

    elif args.dataset == "mn40":
        seen_classnames = [
            "airplane",
            "flower_pot",
            "glass_box",
            "keyboard",
            "monitor",
            "night_stand",
            "sink",
            "table",
        ]
        data_dir = "/data/cd/data/3dor/OS-MN40-core"
        train_dataset = MN40CoreDataset(
            data_dir, "train", modality=args.modality, n_view=args.n_view
        )
        query_dataset = MN40CoreDataset(
            data_dir, "query", modality=args.modality, n_view=args.n_view
        )
        target_dataset = MN40CoreDataset(
            data_dir, "target", modality=args.modality, n_view=args.n_view
        )

    elif args.dataset == "abo":
        seen_classnames = ["mirror", "plant or flower pot", "table", "tent"]
        data_dir = "/data/cd/data/3dor/OS-ABO-core"
        train_dataset = ABOCoreDataset(
            data_dir, "train", modality=args.modality, n_view=args.n_view
        )
        query_dataset = ABOCoreDataset(
            data_dir, "query", modality=args.modality, n_view=args.n_view
        )
        target_dataset = ABOCoreDataset(
            data_dir, "target", modality=args.modality, n_view=args.n_view
        )
    else:
        raise NotImplementedError

    # Set a smaller batch size to reduce GPU memory usage
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0
    )
    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    point_model = build_uni3d_branch(args)

    run_lora(
        args,
        clip_model,
        logit_scale,
        seen_classnames,
        train_loader,
        query_loader,
        target_loader,
        point_model=point_model,
    )


if __name__ == "__main__":

    all_st = time.time()
    main()
    all_sec = time.time() - all_st
    print(
        f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes {all_sec%60:.2f}s!"
    )
    print("All done!")
