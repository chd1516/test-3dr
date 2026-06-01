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


# the first 17 classes are seen classes, the rest are unseen classes
# seen will be used for training, query and target will be used for testing (all samples are unseen classes)
Categories2IDS = {
    "flat-thin wallcomponents___clips": 0,
    "flat-thin wallcomponents___contact switches": 1,
    "flat-thin wallcomponents___miscellaneous": 2,
    "flat-thin wallcomponents___slender thin plates": 3,
    "rectangular-cubic prism___bearing blocks": 4,
    "rectangular-cubic prism___contoured surfaces": 5,
    "rectangular-cubic prism___l blocks": 6,
    "rectangular-cubic prism___machined blocks": 7,
    "rectangular-cubic prism___motor bodies": 8,
    "rectangular-cubic prism___rocker arms": 9,
    "solid of revolution___90 degree elbows": 10,
    "solid of revolution___container like parts": 11,
    "solid of revolution___intersecting pipes": 12,
    "solid of revolution___non-90 degree elbows": 13,
    "solid of revolution___oil pans": 14,
    "solid of revolution___posts": 15,
    "solid of revolution___spoked wheels": 16,
    "flat-thin wallcomponents___bracket like parts": 17,
    "flat-thin wallcomponents___thin plates": 18,
    "rectangular-cubic prism___handles": 19,
    "rectangular-cubic prism___long machine elements": 20,
    "rectangular-cubic prism___machined plates": 21,
    "rectangular-cubic prism___miscellaneous": 22,
    "rectangular-cubic prism___prismatic stock": 23,
    "rectangular-cubic prism___slender links": 24,
    "rectangular-cubic prism___small machined blocks": 25,
    "rectangular-cubic prism___t shaped parts": 26,
    "rectangular-cubic prism___thick plates": 27,
    "rectangular-cubic prism___thick slotted plates": 28,
    "rectangular-cubic prism___u shaped parts": 29,
    "solid of revolution___bearing like parts": 30,
    "solid of revolution___bolt like parts": 31,
    "solid of revolution___cylindrical parts": 32,
    "solid of revolution___discs": 33,
    "solid of revolution___flange like parts": 34,
    "solid of revolution___gear like parts": 35,
    "solid of revolution___long pins": 36,
    "solid of revolution___miscellaneous": 37,
    "solid of revolution___nuts": 38,
    "solid of revolution___pulley like parts": 39,
    "solid of revolution___round change at end": 40,
}


class ESBCoreDataset(Dataset):
    def __init__(self, data_dir, split, modality="mv", n_view=24):
        assert split in ["train", "query", "target"]
        assert modality in ["mv", "vox", "point", "mv_point"]

        self.data_dir = data_dir
        self.split = split
        self.modal = modality
        self.n_view = n_view

        self.samples, self.label_list = self.load_data()
        self.label2idx = {
            label: Categories2IDS[label] for _, label in enumerate(self.label_list)
        }
        self.idx2label = {
            Categories2IDS[label]: label for _, label in enumerate(self.label_list)
        }
        self.num_classes = len(self.label_list)

        if split == "train":
            # transform
            if self.modal in ["mv", "mv_point"]:
                self.img_size = 224
                self.transform = T.Compose(
                    [
                        T.RandomResizedCrop(self.img_size),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                    ]
                )

            if self.modal == "mv_point":
                self.pc_transform = T.Compose(
                    [
                        PointsToTensor(),
                        PointCloudScaling(scale=[0.9, 1.1]),
                        PointCloudCenterAndNormalize(gravity_dim=1),
                        PointCloudRotation(angle=[0.0, 1.0, 0.0]),
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
            # query and target: both has seen and unseen classes
            # import pdb; pdb.set_trace()
            if self.modal in ["mv", "mv_point"]:
                self.img_size = 224
                self.transform = T.Compose(
                    [
                        T.Resize(self.img_size),
                        T.ToTensor(),
                    ]
                )

            if self.modal == "mv_point":
                self.pc_transform = T.Compose(
                    [
                        PointsToTensor(),
                        PointCloudCenterAndNormalize(gravity_dim=1),
                    ]
                )
            elif self.modal == "point":
                self.transform = T.Compose(
                    [
                        PointsToTensor(),
                        PointCloudCenterAndNormalize(gravity_dim=1),
                    ]
                )
            elif self.modal == "vox":
                print("voxel doest not need any transformation")
        else:
            raise NotImplementedError

    def __fetch_img_list(self, instance_path, n_view=24):
        all_filenames = sorted(list(Path(instance_path).glob("image/h_*.jpg")))
        all_view = len(all_filenames)
        filenames = all_filenames[:: all_view // self.n_view][: self.n_view]
        return filenames

    def __fetch_pt_path(self, instance_path, n_pt):
        return Path(instance_path) / "pointcloud" / f"pt_{n_pt}.pts"

    def __fetch_vox_path(self, instance_path, d_vox):
        return Path(instance_path) / "voxel" / f"vox_{d_vox}.ply"

    def __read_vox(self, vox_path, d_vox):
        vox_3d = o3d.io.read_voxel_grid(str(vox_path))
        vox_idx = torch.from_numpy(
            np.array([v.grid_index - 1 for v in vox_3d.get_voxels()])
        ).long()
        vox = torch.zeros((d_vox, d_vox, d_vox))
        vox[vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]] = 1
        return vox.unsqueeze(0)

    def __read_images(self, path_list):
        imgs = [Image.open(v).convert("RGB") for v in path_list]
        return imgs

    def __read_pointcloud(self, pc_path):
        pt = np.asarray(o3d.io.read_point_cloud(str(pc_path)).points)
        pt = pt - np.expand_dims(np.mean(pt, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(pt**2, axis=1)), 0)
        pt = pt / dist
        return pt

    def load_data(self):
        if self.split == "train":
            ############################################################################
            # NOTE: all the samples for training belong to seen classes (17)
            ############################################################################
            # import pdb; pdb.set_trace()
            data_root = Path(self.data_dir)
            train_list, seen_label = [], []
            for label_root in data_root.glob("train/*"):
                label_name = label_root.name
                for obj_path in label_root.glob("*/"):
                    train_list.append({"path": str(obj_path), "label": label_name})
                seen_label.append(label_name)
            seen_label = sorted(set(seen_label))
            return train_list, seen_label

        elif self.split == "query" or self.split == "target":
            ############################################################################
            # NOTE: all the samples fro query and target belong to unseen classes (24)
            # TODO: should we consider the seen classes as well? Maybe ModelNet40 supports this setup
            #       Check it later.
            ############################################################################
            # import pdb; pdb.set_trace()
            split_file_path = Path(self.data_dir) / f"{self.split}_label.txt"

            label_list = []
            sample_list = []
            with open(split_file_path, "r") as fp:
                for line in fp.readlines():
                    obj_name, label_name = line.strip().split(",")
                    sample_list.append(
                        {
                            "path": str(Path(self.data_dir) / self.split / obj_name),
                            "label": label_name,
                        }
                    )
                    label_list.append(label_name)
            return sample_list, sorted(set(label_list))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        instance_path, label = sample["path"], sample["label"]

        if self.modal == "mv_point":
            img_list = self.__fetch_img_list(instance_path, 24)
            imgs = self.__read_images(img_list)
            imgs = [self.transform(img) for img in imgs]
            imgs = torch.stack(imgs)

            pc = self.__read_pointcloud(self.__fetch_pt_path(instance_path, 1024))
            if self.split == "train":
                np.random.shuffle(pc)
            pc = self.pc_transform(pc)

            label = self.label2idx[label]
            return imgs, pc, label, instance_path  # 返回多视图、点云、标签和路径

        if self.modal == "mv":
            img_list = self.__fetch_img_list(instance_path, 24)
            imgs = self.__read_images(img_list)
            imgs = [self.transform(img) for img in imgs]
            imgs = torch.stack(imgs)

            label = self.label2idx[label]
            return imgs, label, instance_path  # 返回图像数据、标签和路径

        elif self.modal == "vox":
            vox = self.__read_vox(self.__fetch_vox_path(instance_path, 32), 32)
            label = self.label2idx[label]
            return vox, label, instance_path  # 返回体素数据、标签和路径

        elif self.modal == "point":
            pc = self.__read_pointcloud(self.__fetch_pt_path(instance_path, 1024))
            if self.split == "train":
                np.random.shuffle(pc)
            pc = self.transform(pc)
            label = self.label2idx[label]
            return pc, label, instance_path  # 返回点云数据、标签和路径
