from dataclasses import dataclass, field
from typing import List
from omegaconf import MISSING


@dataclass
class DataConfig:
    name: str = MISSING
    num_workers: int = 8
    # num views
    num_views: int = MISSING


@dataclass
class PolyMNISTDataConfig(DataConfig):
    num_views: int = 3
    dir_data_base: str = "/usr/scratch/projects/multimodality/data"
    dir_clfs_base: str = (
        "/usr/scratch/projects/multimodality/mvvae/experiments/trained_clfs/PolyMNIST"
    )
    n_clfs_outputs: int = 10
    num_labels: int = 1


@dataclass
class PMtranslatedData75Config(PolyMNISTDataConfig):
    name: str = "PM_translated75"
    suffix_data_train: str = "PolyMNIST_translated_scale075/train"
    suffix_data_test: str = "PolyMNIST_translated_scale075/test"
    suffix_clfs: str = "translated75_resnet"


@dataclass
class CelebADataConfig(DataConfig):
    name: str = "celeba"
    num_views: int = 2
    dir_data: str = "/usr/scratch/projects/multimodality/data/CelebA"
    dir_alphabet: str = (
        "/home/thomas/polybox2/PhD/projects/research_stay/code/mvvae/utils"
    )
    dir_clf: str = (
        "/usr/scratch/projects/multimodality/mvvae/experiments/trained_clfs/CelebA"
    )

    len_sequence: int = 256
    random_text_ordering: bool = False
    random_text_startindex: bool = True
    img_size: int = 64
    image_channels: int = 3
    crop_size_img: int = 148
    n_clfs_outputs: int = 40
    num_labels: int = 40

    num_features: int = 41  # len(alphabet)
    num_layers_img: int = 5
    filter_dim_img: int = 64
    filter_dim_text: int = 64
    beta_img: float = 1.0
    beta_text: float = 1.0
    skip_connections_img_weight_a: float = 1.0
    skip_connections_img_weight_b: float = 1.0
    skip_connections_text_weight_a: float = 1.0
    skip_connections_text_weight_b: float = 1.0

    use_rec_weight: bool = True
    include_channels_rec_weight: bool = False


@dataclass
class CUBDataConfig(DataConfig):
    name: str = "CUB"
    num_views: int = 2
    dir_data: str = "/usr/scratch/projects/multimodality/data/cub"
    num_labels: int = 6
    dir_clf: str = (
        "/usr/scratch/projects/multimodality/mvvae/experiments/trained_clfs/cub"
    )
    beta_img: float = 1.0
    beta_text: float = 1.0
    len_sequence: int = 32
    img_size: int = 64
    n_clfs_outputs: int = 6
    label_names: List[str] = field(
        default_factory=lambda: [
            "blue2red",
            "brown",
            "grey",
            "yellow",
            "black",
            "white",
        ]
    )
