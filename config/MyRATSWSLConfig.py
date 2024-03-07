from dataclasses import dataclass, field
from typing import List, Dict, Optional

from omegaconf import MISSING


@dataclass
class LogConfig:
    # wandb
    wandb_entity: str = "ym2696"
    wandb_group: str = "mw_wsl"
    wandb_run_name: str = "mw_wsl"
    wandb_project_name: str = "mw_wsl"
    wandb_log_freq: int = 50
    wandb_offline: bool = False

    # logs
    dir_logs: str = "/home/mengy13/mvvae/experiments"


@dataclass
class ModelConfig:
    device: str = "cuda"
    batch_size: int = 128
    lr: float = 0.001
    epochs: int = 1000

    latent_dim: int = 2

    resample_eval: bool = False

    # loss hyperparameters
    beta: float = 1.0

    # weight on N(0,1) in mixed prior
    stdnormweight: float = 0.0

    # network architectures
    # use_resnets: bool = True


@dataclass
class EvalConfig:
    # latent representation
    num_samples_train: int = 10000
    max_iteration: int = 10000
    eval_downstream_task: bool = True

    # coherence
    coherence: bool = True

@dataclass
class JointModelConfig(ModelConfig):
    name: str = "joint"


@dataclass
class MixedPriorModelConfig(ModelConfig):
    name: str = "mixedprior"

@dataclass
class DataConfig:
    name: str = MISSING
    num_workers: int = 8
    # num views
    num_views: int = MISSING 
    dir_clfs_base: str = "/home/mengy13/mvvae/experiments/trained_clfs/rats"

@dataclass
class SPIKEDataConfig(DataConfig):
    name: str = "SPIKE"
    num_views: int = 5
    dir_data: str = "/home/mengy13/rats"
    suffix_clfs: str = "spike"

@dataclass
class MyRATSWSLConfig:
    seed: int = 0
    checkpoint_metric: str = "val/loss/loss"
    # logger
    log: LogConfig = MISSING
    # dataset
    dataset: DataConfig = MISSING
    # model
    model: ModelConfig = MISSING
    # eval
    eval: EvalConfig = MISSING
