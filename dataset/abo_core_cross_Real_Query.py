import os
import random
import itertools

import glob
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import open3d as o3d
from .pc_transforms import (
    PointsToTensor,
    PointCloudScaling,
    PointCloudCenterAndNormalize,
    PointCloudRotation,
)
import torch.utils.data as data
import torch.distributed as dist


# the first 4 classes are seen classes, the rest are unseen classes
# seen will be used for training, query and target will be used for testing (all samples are unseen classes)
Categories2IDS = {
    "mirror": 0,
    "plant or flower pot": 1,
    "table": 2,
    "tent": 3,
    "bed": 4,
    "bench": 5,
    "cabinet": 6,
    "cart": 7,
    "chair": 8,
    "container or basket": 9,
    "dresser": 10,
    "exercise weight": 11,
    "fan": 12,
    "ladder": 13,
    "lamp": 14,
    "ottoman": 15,
    "picture frame or painting": 16,
    "pillow": 17,
    "shelf": 18,
    "sofa": 19,
    "vase": 20,
}


class ABOCoreDataset(Dataset):
    def __init__(self, data_dir, split, modality="mv", n_view=24):
        super(ABOCoreDataset, self).__init__()
        assert split in ["train", "query", "target"]
        assert modality in ["mv", "vox", "point"]

        self.data_dir = data_dir
        self.split = split
        self.modal = modality
        self.samples, self.label_list = self.load_data()
        self.label2idx = {
            label: Categories2IDS[label] for _, label in enumerate(self.label_list)
        }
        self.idx2label = {
            Categories2IDS[label]: label for _, label in enumerate(self.label_list)
        }
        self.num_classes = len(self.label_list)
        self.n_view = n_view

        if split == "train":
            # transform
            if self.modal == "mv":
                self.img_size = 224
                self.transform = T.Compose(
                    [
                        T.RandomResizedCrop(self.img_size),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                    ]
                )

            elif self.modal == "point":
                # import pdb; pdb.set_trace()
                # transform pc to tensor
                self.transform = T.Compose(
                    [
                        PointsToTensor(),
                        PointCloudScaling(scale=[0.9, 1.1]),
                        PointCloudCenterAndNormalize(gravity_dim=1),
                        PointCloudRotation(angle=[0.0, 1.0, 0.0]),
                    ]
                )

        elif split == "query" or split == "target":
            if self.modal == "mv":
                self.img_size = 224
                self.transform = T.Compose(
                    [
                        T.Resize(256),  # 先缩放
                        T.CenterCrop(self.img_size),  # 裁剪为 224x224
                        T.ToTensor(),
                    ]
                )
        else:
            raise NotImplementedError

    def __fetch_img_list(self, instance_path):
        all_filenames = sorted(
            list(Path(instance_path).glob("real_image/main_*.jpg")),
        )
        all_view = len(all_filenames)
        # import pdb; pdb.set_trace()
        filenames = all_filenames[:: all_view // self.n_view][: self.n_view]
        return filenames

    def __read_images(self, path_list):
        imgs = [Image.open(v).convert("RGB") for v in path_list]
        return imgs

    def load_data(self):
        if self.split == "query" or self.split == "target":
            split_file_path = f"splits/mn40-abo/{self.split}.txt"
            label_list = []
            sample_list = []
            with open(split_file_path, "r") as fp:
                for line in fp.readlines():
                    obj_name, label_name = line.strip().split(",")
                    sample_list.append(
                        {
                            "path": str(Path(self.data_dir) / obj_name),
                            "label": label_name,
                        }
                    )
                    label_list.append(label_name)
            return sample_list, sorted(set(label_list))
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        instance_path, label = sample["path"], sample["label"]

        if self.modal == "mv":
            img_list = self.__fetch_img_list(instance_path)
            imgs = self.__read_images(img_list)
            imgs = [self.transform(img) for img in imgs]
            imgs = torch.stack(imgs)

            label = self.label2idx[label]
            return imgs, label, instance_path  # 返回图像数据、标签和路径
