from dataclasses import dataclass, field
from typing import List, Dict, Optional

from omegaconf import MISSING

@dataclass
class LogConfig:
    # wandb
    wandb_entity: str = "ym2696"
    wandb_group: str = "SPIKEclf"
    wandb_run_name: str = "clf"
    wandb_project_name: str = "mw_wsl_clf"
    wandb_log_freq: int = 50
    wandb_offline: bool = False

    # logs
    dir_logs: str = "/home/mengy13/mvvae/experiments/clfs"


@dataclass
class DataConfig:
    name: str = MISSING
    num_workers: int = 8
    # num views
    num_views: int = 5 
    dir_clfs_base: str = "/home/mengy13/mvvae/experiments/trained_clfs/rats"

@dataclass
class SPIKEDataConfig(DataConfig):
    name: str = "SPIKE"
    dir_data: str = "/home/mengy13/rats"
    suffix_clfs: str = "spike"


@dataclass
class ModelConfig:
    device: str = "cuda"
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 51 


@dataclass
class MyClfRatsConfig:
    seed: int = 0
    checkpoint_metric: str = "val/loss/mean_acc"
    model: ModelConfig = MISSING
    log: LogConfig = MISSING
    dataset: DataConfig = MISSING
