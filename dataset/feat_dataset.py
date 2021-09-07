import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from sklearn.model_selection import StratifiedKFold


def set_fold(labels, n_split):
    skf = StratifiedKFold(n_splits=n_split)
    df_folds = pd.DataFrame({"labels": labels})
    df_folds.loc[:, "fold"] = 0
    for fold_number, (train_index, val_index) in enumerate(
        skf.split(X=df_folds.index, y=df_folds.labels)
    ):
        df_folds.loc[df_folds.iloc[val_index].index, "fold"] = fold_number
    return df_folds


class ImgDataset(data.Dataset):
    def __init__(
        self,
        mode="train",
        data_dir="./data",
        transform=None,
        fold_num=0,
    ):
        self._data_dir = data_dir
        if mode == "train" or mode == "valid":
            self._labels = pd.read_csv(Path("train.csv"))["labels"]
            self._data_dir = self._data_dir / Path("train")
            self._labels = set_fold(self._labels, n_split=5)
            self._labels = self._labels.reset_index().rename(
                columns={"index": "img_name"}
            )
            if mode == "train":
                self._labels = self._labels[self._labels.fold != fold_num]
            elif mode == "valid":
                self._labels = self._labels[self._labels.fold == fold_num]

        else:
            self._data_dir = self._data_dir / Path("test")

        self._transform = transform
        self._mode = mode

    def __getitem__(self, idx):
        if self._mode == "test":
            img_name = idx
        else:
            img_name = self._labels.img_name.values[idx]

        image_dir = self._data_dir / Path(str(img_name) + ".jpg")
        image = pil_loader(image_dir)
        if self._transform:
            image = self._transform(image)

        if self._mode == "train" or self._mode == "valid":
            label = torch.tensor(self._labels.labels.values[idx])
            return image, label
        elif self._mode == "test":
            return image

    def __len__(self):
        if self._mode == "test":
            return 500
        else:
            return len(self._labels)


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


if __name__ == "__main__":
    d = ImgDataset(mode="valid")
    for img, label in d:
        print(label)
