import os, sys
import json
import torch
from torchvision import transforms
import PIL.Image as Image

from utils.PolyMNISTDataset import PolyMNIST
from utils.CelebADataset import CelebADataset

transform = transforms.Compose([transforms.ToTensor()])


def get_dataset(cfg):
    if cfg.dataset.name.startswith("PM"):
        ds = get_dataset_PM(cfg)
    elif cfg.dataset.name.startswith("celeba"):
        ds = get_dataset_celeba(cfg)
    else:
        print("dataset unknown...exit")
        sys.exit()
    return ds


def get_dataset_PM(cfg):
    dir_data_train = os.path.join(
        cfg.dataset.dir_data_base, cfg.dataset.suffix_data_train
    )
    train_dst = PolyMNIST(dir_data_train, cfg.dataset.num_views, transform=transform)
    dir_data_test = os.path.join(
        cfg.dataset.dir_data_base, cfg.dataset.suffix_data_test
    )
    val_dst = PolyMNIST(dir_data_test, cfg.dataset.num_views, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dst,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dst,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
    )
    return train_loader, train_dst, val_loader, val_dst


def get_dataset_celeba(cfg):
    transform = get_transform_celeba(cfg)
    alphabet_path = os.path.join(cfg.dataset.dir_alphabet, "alphabet.json")
    with open(alphabet_path) as alphabet_file:
        alphabet = str("".join(json.load(alphabet_file)))

    d_train = CelebADataset(cfg, alphabet, partition=0, transform=transform)
    d_eval = CelebADataset(cfg, alphabet, partition=1, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        d_train,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        d_eval,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
    )
    return train_loader, d_train, val_loader, d_eval


def get_transform_celeba(cfg):
    offset_height = (218 - cfg.dataset.crop_size_img) // 2
    offset_width = (178 - cfg.dataset.crop_size_img) // 2
    crop = lambda x: x[
        :,
        offset_height : offset_height + cfg.dataset.crop_size_img,
        offset_width : offset_width + cfg.dataset.crop_size_img,
    ]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(crop),
            transforms.ToPILImage(),
            transforms.Resize(
                size=(cfg.dataset.img_size, cfg.dataset.img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ]
    )

    return transform
