from typing import Any
import os
import shutil

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg


def normalize_tin_validation_folder_structure(
    path: str,
    images_folder: str = "images",
    annotations_file: str = "val_annotations.txt",
) -> None:
    # Check if files/annotations are still there to see
    # if we already run reorganize the folder structure.
    images_folder = os.path.join(path, images_folder)
    annotations_file = os.path.join(path, annotations_file)

    # Exists
    if not os.path.exists(images_folder) and not os.path.exists(annotations_file):
        if not os.listdir(path):
            raise RuntimeError("Validation folder is empty.")
        return

    # Parse the annotations
    with open(annotations_file) as f:
        for line in f:
            values = line.split()
            img = values[0]
            label = values[1]
            img_file = os.path.join(images_folder, values[0])
            label_folder = os.path.join(path, label)
            os.makedirs(label_folder, exist_ok=True)
            try:
                shutil.move(img_file, os.path.join(label_folder, img))
            except FileNotFoundError:
                continue

    os.sync()
    assert not os.listdir(images_folder)
    shutil.rmtree(images_folder)
    os.remove(annotations_file)
    os.sync()


class TinyImageNet(ImageFolder):
    """Dataset for TinyImageNet-200

    Taken from: https://gist.github.com/lromor/bcfc69dcf31b2f3244358aea10b7a11b
    """

    base_folder = "tiny-imagenet-200"
    zip_md5 = "90528d7ca1a48142e341f4ef8d21d0de"
    splits = ("train", "val")
    filename = "tiny-imagenet-200.zip"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    def __init__(
        self, root: str, split: str = "train", download: bool = False, **kwargs: Any
    ) -> None:
        self.data_root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", self.splits)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )
        super().__init__(self.split_folder, **kwargs)

    @property
    def dataset_folder(self) -> str:
        return os.path.join(self.data_root, self.base_folder)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.dataset_folder, self.split)

    def _check_exists(self) -> bool:
        return os.path.exists(self.split_folder)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

    def download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(
            self.url,
            self.data_root,
            filename=self.filename,
            remove_finished=True,
            md5=self.zip_md5,
        )
        assert "val" in self.splits
        normalize_tin_validation_folder_structure(os.path.join(self.dataset_folder, "val"))
