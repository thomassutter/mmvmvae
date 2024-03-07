import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import wandb

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

from utils import dataset_rats
from utils.eval import load_modality_clfs

from config.MyRATSWSLConfig import MyRATSWSLConfig
from config.MyRATSWSLConfig import LogConfig
from config.MyRATSWSLConfig import ModelConfig
from config.MyRATSWSLConfig import JointModelConfig
from config.MyRATSWSLConfig import MixedPriorModelConfig
from config.MyRATSWSLConfig import DataConfig
from config.MyRATSWSLConfig import SPIKEDataConfig
from config.MyRATSWSLConfig import EvalConfig


from mv_vaes.spike_mixedprior_vae import SPIKEMixedPriorVAE as SPIKEMixedPriorVAE
from mv_vaes.spike_joint_vae import SPIKEJointVAE as SPIKEJointVAE

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(group="log", name="log", node=LogConfig)
cs.store(group="model", name="joint", node=JointModelConfig)
cs.store(group="model", name="mixedprior", node=MixedPriorModelConfig)
cs.store(group="eval", name="eval", node=EvalConfig)
cs.store(group="dataset", name="SPIKE", node=SPIKEDataConfig)
cs.store(group="dataset", name="dataset", node=DataConfig)
cs.store(name="base_config", node=MyRATSWSLConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: MyRATSWSLConfig):
    print(cfg)
    pl.seed_everything(cfg.seed, workers=True)

    # init model
    model = None

    # get data loaders and specify model

    # SPIKE data
    if cfg.dataset.name == "SPIKE":
        train_loader, train_dst, val_loader, val_dst = dataset_rats.get_dataset(cfg)
        if cfg.model.name == "joint":
            model = SPIKEJointVAE(cfg)
        elif cfg.model.name == "mixedprior":
            model = SPIKEMixedPriorVAE(cfg)

    assert model is not None

    summary = ModelSummary(model, max_depth=2)
    print(summary)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.log.dir_logs,
        monitor=cfg.checkpoint_metric,
        mode="min",
        save_last=True,
    )
    wandb_logger = WandbLogger(
        name=cfg.log.wandb_run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        project=cfg.log.wandb_project_name,
        group=cfg.log.wandb_group,
        offline=cfg.log.wandb_offline,
        entity=cfg.log.wandb_entity,
        save_dir=cfg.log.dir_logs,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.model.epochs,
        devices=1,
        accelerator="gpu" if cfg.model.device == "cuda" else cfg.model.device,
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        deterministic=True,
        callbacks=[checkpoint_callback],
    )

    trainer.logger.watch(model, log="all")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    for m in range(cfg.dataset.num_views):
        model.logger.log_metrics(
            {f"final_scores/downstream_lr/acc_v_%2d" % m: model.final_accuracies_lr[m]}
        )
    for m in range(cfg.dataset.num_views):
        model.logger.log_metrics(
            {f"final_scores/downstream_lr/acc_one_clf_v_%2d" % m: model.final_accuracies_lr_one_clf[m]}
        )
    for m in range(cfg.dataset.num_views):
        for m_tilde in range(cfg.dataset.num_views):
            model.logger.log_metrics(
                {f"final_scores/coherence/rec_loss_v_%1d_to_%1d" % (m, m_tilde): model.final_rec_loss_coh[m*cfg.dataset.num_views + m_tilde]}
            )
            model.logger.log_metrics(
                {f"final_scores/coherence/acc_v_%1d_to_%1d" % (m, m_tilde): model.final_accuracies_coh[m*cfg.dataset.num_views + m_tilde]}
            )

if __name__ == "__main__":
    run_experiment()
