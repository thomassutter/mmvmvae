import os
import torch
from torchvision import transforms

from utils.RatsDataset import LFP, SPIKE

def get_dataset(cfg):
    dir_data = os.path.join(cfg.dataset.dir_data)
    if cfg.dataset.name == "LFP":
        train_dst = LFP(
            dir_data=dir_data, train=True, num_views=cfg.dataset.num_views
        )
        val_dst = LFP(
            dir_data=dir_data, train=False, num_views=cfg.dataset.num_views
        )
    elif cfg.dataset.name == "SPIKE":
        train_dst = SPIKE(
            dir_data=dir_data, train=True, num_views=cfg.dataset.num_views
        )
        val_dst = SPIKE(
            dir_data=dir_data, train=False, num_views=cfg.dataset.num_views
        )
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
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
    )
    return train_loader, train_dst, val_loader, val_dst
