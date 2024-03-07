from dataclasses import dataclass


@dataclass
class ModelConfig:
    device: str = "cuda"
    batch_size: int = 128
    lr: float = 5e-4
    epochs: int = 500

    latent_dim: int = 256

    resample_eval: bool = False

    # loss hyperparameters
    beta: float = 1.0

    # network architectures
    use_resnets: bool = True

    # annealing
    temp_annealing: str = "exp"


@dataclass
class JointModelConfig(ModelConfig):
    name: str = "joint"
    aggregation: str = "poe"


@dataclass
class MixedPriorModelConfig(ModelConfig):
    name: str = "mixedprior"
    drpm_prior: bool = False

    # weight on N(0,1) in mixed prior
    alpha_annealing: bool = True
    init_alpha_value: float = 1.0
    final_alpha_value: float = 0.0
    alpha_annealing_steps: int = 300000


@dataclass
class UnimodalModelConfig(ModelConfig):
    name: str = "unimodal"
