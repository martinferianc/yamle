"""
author: Mihai Suteu
date: 15/05/19

Adapted from: https://github.com/xapharius/pytorch-nyuv2/
"""

import os
import h5py
import shutil
import tarfile
import zipfile
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url


class NYUv2(Dataset):
    """
    PyTorch wrapper for the NYUv2 dataset.
    Data sources available: Semantic Segmentation and Depth Estimation.

    1. Semantic Segmentation: 1 channel representing one of the 14 (0 -
    background) classes.

    2. Depth Images: 1 channel with floats.

    Args:
        root (str): Root directory of dataset where ``NYUv2/processed/training.pt``
            and  ``NYUv2/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
        download (bool, optional): If true, downloads the dataset from the internet and
        task (str, optional): Choose from: segmentation, depth.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        task: str = "segmentation",
    ) -> None:
        super().__init__()
        self.root = root

        assert task in [
            "segmentation",
            "depth",
        ], f"Task {task} not supported. Choose from: segmentation, depth"
        self.task = task

        self.train = train
        self._split = "train" if train else "test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not complete." + " You can use download=True to download it"
            )

        # rgb folder as ground truth
        self._files = sorted(os.listdir(os.path.join(root, f"{self._split}_rgb")))

    def __getitem__(self, index: int):
        def folder(name):
            return os.path.join(self.root, f"{self._split}_{name}")

        x = Image.open(os.path.join(folder("rgb"), self._files[index]))

        if self.task == "segmentation":
            y = Image.open(os.path.join(folder("seg13"), self._files[index]))
        elif self.task == "depth":
            y = Image.open(os.path.join(folder("depth"), self._files[index]))
        else:
            raise NotImplementedError(f"Task {self.task} not implemented")
        return x, y

    def __len__(self):
        return len(self._files)

    def _check_exists(self) -> bool:
        """
        Only checking for folder existence
        """
        try:
            for split in ["train", "test"]:
                part = "seg13" if self.task == "segmentation" else "depth"
                path = os.path.join(self.root, f"{split}_{part}")
                if not os.path.exists(path):
                    raise FileNotFoundError("Missing Folder")
        except FileNotFoundError as e:
            return False
        return True

    def download(self):
        if self._check_exists():
            return
        download_rgb(self.root)
        if self.task == "segmentation":
            download_seg(self.root)
        if self.task == "depth":
            download_depth(self.root)


def download_rgb(root: str):
    train_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz"
    test_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz"

    def _proc(url: str, dst: str):
        if not os.path.exists(dst):
            tar = os.path.join(root, url.split("/")[-1])
            if not os.path.exists(tar):
                download_url(url, root)
            if os.path.exists(tar):
                _unpack(tar)
                _replace_folder(tar.rstrip(".tgz"), dst)
                _rename_files(dst, lambda x: x.split("_")[2])

    _proc(train_url, os.path.join(root, "train_rgb"))
    _proc(test_url, os.path.join(root, "test_rgb"))


def download_seg(root: str):
    train_url = "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/train_labels_13/nyuv2_train_class13.tgz"
    test_url = "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/test_labels_13/nyuv2_test_class13.tgz"

    def _proc(url: str, dst: str):
        if not os.path.exists(dst):
            tar = os.path.join(root, url.split("/")[-1])
            if not os.path.exists(tar):
                download_url(url, root)
            if os.path.exists(tar):
                _unpack(tar)
                _replace_folder(tar.rstrip(".tgz"), dst)
                _rename_files(dst, lambda x: x.split("_")[3])

    _proc(train_url, os.path.join(root, "train_seg13"))
    _proc(test_url, os.path.join(root, "test_seg13"))


def download_depth(root: str):
    url = (
        "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    )
    train_dst = os.path.join(root, "train_depth")
    test_dst = os.path.join(root, "test_depth")

    if not os.path.exists(train_dst) or not os.path.exists(test_dst):
        tar = os.path.join(root, url.split("/")[-1])
        if not os.path.exists(tar):
            download_url(url, root)
        if os.path.exists(tar):
            train_ids = [
                f.split(".")[0] for f in os.listdir(os.path.join(root, "train_rgb"))
            ]
            _create_depth_files(tar, root, train_ids)


def _unpack(file: str):
    """
    Unpacks tar and zip, does nothing for any other type
    :param file: path of file
    """
    path = file.rsplit(".", 1)[0]

    if file.endswith(".tgz"):
        tar = tarfile.open(file, "r:gz")
        tar.extractall(path)
        tar.close()
    elif file.endswith(".zip"):
        zip = zipfile.ZipFile(file, "r")
        zip.extractall(path)
        zip.close()


def _rename_files(folder: str, rename_func: callable):
    """
    Renames all files inside a folder based on the passed rename function
    :param folder: path to folder that contains files
    :param rename_func: function renaming filename (not including path) str -> str
    """
    imgs_old = os.listdir(folder)
    imgs_new = [rename_func(file) for file in imgs_old]
    for img_old, img_new in zip(imgs_old, imgs_new):
        shutil.move(os.path.join(folder, img_old), os.path.join(folder, img_new))


def _replace_folder(src: str, dst: str):
    """
    Rename src into dst, replacing/overwriting dst if it exists.
    """
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)


def _create_depth_files(mat_file: str, root: str, train_ids: list):
    """
    Extract the depth arrays from the mat file into images
    :param mat_file: path to the official labelled dataset .mat file
    :param root: The root directory of the dataset
    :param train_ids: the IDs of the training images as string (for splitting)
    """
    os.mkdir(os.path.join(root, "train_depth"))
    os.mkdir(os.path.join(root, "test_depth"))
    train_ids = set(train_ids)

    depths = h5py.File(mat_file, "r")["depths"]
    for i in range(len(depths)):
        img = (depths[i] * 1e4).astype(np.uint16).T
        id_ = str(i + 1).zfill(4)
        folder = "train" if id_ in train_ids else "test"
        save_path = os.path.join(root, f"{folder}_depth", id_ + ".png")
        Image.fromarray(img).save(save_path)
