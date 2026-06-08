import sys
import argparse
from tqdm import tqdm
import time
import random
import numpy as np
import scipy.spatial
import torch
from misc_utils.metric_tools import map_score, eval_all_metric
from dataset.feat_dataset import FeatDataset
import numpy as np
import torch
import yaml
import torch.nn as nn


def setup_seed():
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print(f"random seed: {seed}")


@torch.no_grad()
def test_model_clip_osr3d_feats(
    args,
    eval_all=False,
):
    save_file_suffix = f"{args.backbone}".replace("/", "_")
    save_file_suffix += f"_{args.r}"
    if args.use_uni3d:
        save_file_suffix += "_uni3d"

    query_feats = np.load(
        f"output/image_feats/{args.dataset}_query_feats_{save_file_suffix}.npy"
    )
    query_labels = np.load(
        f"output/image_feats/{args.dataset}_query_labels_{save_file_suffix}.npy"
    )
    target_feats = np.load(
        f"output/image_feats/{args.dataset}_target_feats_{save_file_suffix}.npy"
    )
    target_labels = np.load(
        f"output/image_feats/{args.dataset}_target_labels_{save_file_suffix}.npy"
    )
    query_feats = torch.tensor(query_feats).cuda()
    query_labels = torch.tensor(query_labels).cuda()
    target_feats = torch.tensor(target_feats).cuda()
    target_labels = torch.tensor(target_labels).cuda()

    query_feats = query_feats.mean(dim=1)
    target_feats = target_feats.mean(dim=1)

    save_file_suffix = save_file_suffix + "_Q" + f"{args.question}"
    query_text = np.load(
        f"output/text_feats/{args.dataset}_query_{save_file_suffix}.npy"
    )
    target_text = np.load(
        f"output/text_feats/{args.dataset}_target_{save_file_suffix}.npy"
    )
    query_text = torch.tensor(query_text).cuda()
    target_text = torch.tensor(target_text).cuda()

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    try:
        fusion_rate = config[args.dataset][args.backbone]["fusion_rate"]
    except KeyError:
        # If fusion_rate is not specified in the config, use the default value 0.5
        fusion_rate = 0.5

    combined_feats_query = query_feats + query_text * fusion_rate
    combined_feats_target = target_feats + target_text * fusion_rate

    tanh = nn.Tanh()

    combined_feats_query = tanh(combined_feats_query)
    combined_feats_target = tanh(combined_feats_target)

    # retrieval evaluation
    print(
        "-------------------------------------- Before fusion --------------------------------------"
    )
    retrieval_eval(query_feats, target_feats, query_labels, target_labels, eval_all)

    print("")
    print(
        "-------------------------------------- After fusion --------------------------------------"
    )
    retrieval_eval(
        combined_feats_query,
        combined_feats_target,
        query_labels,
        target_labels,
        eval_all,
    )


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


def main():
    # args
    parser = argparse.ArgumentParser()
    # abo -> mn40: abo as query, mn40 as target
    parser.add_argument(
        "--dataset",
        type=str,
        default="esb",
        help="esb, ntu, mn40, abo",
    )
    parser.add_argument("--backbone", default="ViT-B/32", type=str)
    parser.add_argument(
        "--r", default=2, type=int, help="the rank of the low-rank matrices"
    )
    parser.add_argument(
        "--zero_shot",
        default=False,
    )
    parser.add_argument("--question", default=1, type=int)
    parser.add_argument(
        "--use_uni3d",
        default=False,
        action="store_true",
        help="Load fused CLIP+Uni3D feature files produced by image_feats.py --use_uni3d.",
    )
    args = parser.parse_args()
    print(args)

    setup_seed()

    test_model_clip_osr3d_feats(
        args,
        eval_all=True,
    )


if __name__ == "__main__":

    all_st = time.time()
    main()
    all_sec = time.time() - all_st
    print(
        f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes {all_sec%60:.2f}s!"
    )
    print("All done!")
