from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class LogConfig:
    # wandb
    wandb_entity: str = "thomassutter"
    wandb_group: str = "MVVAEclf"
    wandb_run_name: str = ""
    wandb_project_name: str = "mvvae_clf"
    wandb_log_freq: int = 50
    wandb_offline: bool = False

    # logs
    dir_logs: str = "/usr/scratch/projects/multimodality/mvvae/experiments/clfs"


@dataclass
class DataConfig:
    name: str = MISSING
    num_workers: int = 8
    # num views
    num_views: int = MISSING


@dataclass
class PolyMNISTConfig(DataConfig):
    num_views: int = 3
    num_workers: int = 8
    dir_data_base: str = "/usr/scratch/projects/multimodality/data"
    dir_clfs_base: str = (
        "/usr/scratch/projects/multimodality/mvvae/experiments/trained_clfs/PolyMNIST"
    )



@dataclass
class PMvanillaDataConfig(PolyMNISTConfig):
    name: str = "PM_vanilla"
    suffix_data_train: str = "PolyMNIST_vanilla/train"
    suffix_data_test: str = "PolyMNIST_vanilla/test"
    suffix_clfs: str = "vanilla_resnet"


@dataclass
class PMtranslatedData50Config(PolyMNISTConfig):
    name: str = "PM_translated_50"
    suffix_data_train: str = "PolyMNIST_translated_50/train"
    suffix_data_test: str = "PolyMNIST_translated_50/test"
    suffix_clfs: str = "translatedl50_resnet"


@dataclass
class PMtranslatedData55Config(PolyMNISTConfig):
    name: str = "PM_translated_55"
    suffix_data_train: str = "PolyMNIST_translated_55/train"
    suffix_data_test: str = "PolyMNIST_translated_55/test"
    suffix_clfs: str = "translatedl55_resnet"


@dataclass
class PMtranslatedData60Config(PolyMNISTConfig):
    name: str = "PM_translated_60"
    suffix_data_train: str = "PolyMNIST_translated_60/train"
    suffix_data_test: str = "PolyMNIST_translated_60/test"
    suffix_clfs: str = "translated60_resnet"


@dataclass
class PMtranslatedData65Config(PolyMNISTConfig):
    name: str = "PM_translated_65"
    suffix_data_train: str = "PolyMNIST_translated_65/train"
    suffix_data_test: str = "PolyMNIST_translated_65/test"
    suffix_clfs: str = "translated65_resnet"


@dataclass
class PMtranslatedData70Config(PolyMNISTConfig):
    name: str = "PM_translated_70"
    suffix_data_train: str = "translated_70/train"
    suffix_data_test: str = "translated_70/test"
    suffix_clfs: str = "translated70_resnet"


@dataclass
class PMtranslatedData75Config(PolyMNISTConfig):
    name: str = "PM_translated75"
    suffix_data_train: str = "PolyMNIST_translated_scale075/train"
    suffix_data_test: str = "PolyMNIST_translated_scale075/test"
    suffix_clfs: str = "translated75_resnet"


@dataclass
class PMtranslatedData50FixedConfig(PolyMNISTConfig):
    name: str = "PM_translated_50_fixed"
    suffix_data_train: str = "PolyMNIST_translated_50_fixed/train"
    suffix_data_test: str = "PolyMNIST_translated_50_fixed/test"
    suffix_clfs: str = "translated_50_fixed_resnet"


@dataclass
class PMrotatedDataConfig(PolyMNISTConfig):
    name: str = "PM_rotated"
    suffix_data_train: str = "PolyMNIST_rotated/train"
    suffix_data_test: str = "PolyMNIST_rotated/test"
    suffix_clfs: str = "rotated_resnet"


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
    random_text_ordering: bool = True
    random_text_startindex: bool = False
    img_size: int = 64
    image_channels: int = 3
    crop_size_img: int = 148
    n_clfs_outputs: int = 40
    num_labels: int = 40

    num_features: int = 41  # len(alphabet)
    num_layers_text: int = 7
    num_layers_img: int = 5
    filter_dim_img: int = 128
    filter_dim_text: int = 128
    skip_connections_weight_a: float = 1.0
    skip_connections_weight_b: float = 1.0

    use_rec_weight: bool = True
    include_channels_rec_weight: bool = True

@dataclass
class SPIKEDataConfig(DataConfig):
    name: str = "SPIKE"
    num_views: int = 5 
    dir_clfs_base: str = "/home/mengy13/mvvae/experiments/trained_clfs/rats"
    dir_data: str = "/home/mengy13/rats"
    suffix_clfs: str = "spike"


@dataclass
class ModelConfig:
    device: str = "cuda"
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 50


@dataclass
class MyClfConfig:
    seed: int = 0
    checkpoint_metric: str = "val/loss/mean_ap"
    model: ModelConfig = MISSING
    log: LogConfig = MISSING
    dataset: DataConfig = MISSING

@dataclass
class MyClfRatsConfig:
    seed: int = 0
    checkpoint_metric: str = "val/loss/mean_acc"
    model: ModelConfig = MISSING
    log: LogConfig = MISSING
    dataset: DataConfig = MISSING
