import os

import pandas as pd
from torch.utils.data import Dataset
import PIL.Image as Image

import torch

from utils import text as text


class CelebADataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, cfg, alphabet, partition=0, transform=None):
        filename_text = os.path.join(
            cfg.dataset.dir_data,
            "list_attr_text_"
            + str(cfg.dataset.len_sequence).zfill(3)
            + "_"
            + str(cfg.dataset.random_text_ordering)
            + "_"
            + str(cfg.dataset.random_text_startindex)
            + "_celeba.csv",
        )
        filename_partition = os.path.join(
            cfg.dataset.dir_data, "list_eval_partition.csv"
        )
        filename_attributes = os.path.join(cfg.dataset.dir_data, "list_attr_celeba.csv")

        df_text = pd.read_csv(filename_text)
        df_partition = pd.read_csv(filename_partition)
        df_attributes = pd.read_csv(filename_attributes)

        self.cfg = cfg
        self.img_dir = os.path.join(cfg.dataset.dir_data, "img_align_celeba")
        self.txt_path = filename_text
        self.attrributes_path = filename_attributes
        self.partition_path = filename_partition

        self.alphabet = alphabet
        self.img_names = df_text.loc[df_partition["partition"] == partition][
            "image_id"
        ].values
        self.attributes = df_attributes.loc[df_partition["partition"] == partition]
        self.labels = df_attributes.loc[df_partition["partition"] == partition].values
        self.label_names = list(df_attributes.columns)[1:]
        self.y = df_text.loc[df_partition["partition"] == partition]["text"].values
        self.transform = transform

    def __getitem__(self, index):
        with Image.open(os.path.join(self.img_dir, self.img_names[index])) as img:
            if self.transform is not None:
                img = self.transform(img)
            text_str = text.one_hot_encode(
                self.cfg.dataset.len_sequence, self.alphabet, self.y[index]
            )
            label = torch.from_numpy((self.labels[index, 1:] > 0).astype(int)).float()
            sample = {"img": img, "text": text_str}
            return sample, label

    def __len__(self):
        return self.y.shape[0]

    def get_text_str(self, index):
        return self.y[index]
