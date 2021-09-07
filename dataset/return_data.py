import torch
from torch.utils.data import DataLoader

from .feat_dataset import ImgDataset
from .transform import return_img_transform


def return_dataset(config, fold_num=0):
    dataset_type = config.type
    transforms = return_img_transform()
    train_dataset = ImgDataset(mode="train", transform=transforms, fold_num=fold_num)
    valid_dataset = ImgDataset(mode="valid", transform=transforms, fold_num=fold_num)
    return train_dataset, valid_dataset


def return_dataloader(config, fold_num):
    train_set, valid_set = return_dataset(config, fold_num)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        # num_workers=8,
        pin_memory=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=1,
        drop_last=False,
        num_workers=0,
    )
    return train_loader, valid_loader
