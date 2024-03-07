from dataclasses import dataclass

from omegaconf import MISSING

from config.DatasetConfig import DataConfig
from config.ModelConfig import ModelConfig


@dataclass
class LogConfig:
    # wandb
    wandb_entity: str = "thomassutter"
    wandb_group: str = "mv_wsl"
    wandb_run_name: str = ""
    wandb_project_name: str = "mvvae"
    wandb_log_freq: int = 50
    wandb_offline: bool = False
    wandb_local_instance: bool = False

    # logs
    dir_logs: str = "/usr/scratch/projects/multimodality/mvvae/experiments"

    # logging frequencies
    downstream_logging_frequency: int = 1
    coherence_logging_frequency: int = 1
    img_plotting_frequency: int = 1

    # debug level wandb
    debug: bool = False


@dataclass
class EvalConfig:
    # latent representation
    num_samples_train: int = 10000
    max_iteration: int = 10000

    # latent representation evaluation
    eval_downstream_task: bool = True
    # coherence
    coherence: bool = True


@dataclass
class MyMVWSLConfig:
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
