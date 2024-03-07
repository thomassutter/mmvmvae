import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import wandb

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from utils import dataset

from config.MyMVWSLConfig import MyMVWSLConfig
from config.MyMVWSLConfig import LogConfig
from config.ModelConfig import JointModelConfig
from config.ModelConfig import MixedPriorModelConfig
from config.ModelConfig import UnimodalModelConfig
from config.DatasetConfig import PMtranslatedData75Config
from config.DatasetConfig import CelebADataConfig
from config.MyMVWSLConfig import EvalConfig

from mv_vaes.mv_joint_vae import MVJointVAE as MVJointVAE
from mv_vaes.mv_unimodal_vae import MVunimodalVAE as MVunimodalVAE
from mv_vaes.mv_mixedprior_vae import MVMixedPriorVAE as MVMixedPriorVAE

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(group="log", name="log", node=LogConfig)
cs.store(group="model", name="joint", node=JointModelConfig)
cs.store(group="model", name="mixedprior", node=MixedPriorModelConfig)
cs.store(group="model", name="unimodal", node=UnimodalModelConfig)
cs.store(group="eval", name="eval", node=EvalConfig)
cs.store(group="dataset", name="PMtranslated75", node=PMtranslatedData75Config)
cs.store(group="dataset", name="CelebA", node=CelebADataConfig)
# cs.store(group="dataset", name="dataset", node=DataConfig)
cs.store(name="base_config", node=MyMVWSLConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: MyMVWSLConfig):
    print(cfg)
    if cfg.log.wandb_local_instance:
        wandb.login(host=os.getenv("WANDB_LOCAL_URL"))
    else:
        wandb.login(host="https://api.wandb.ai")
    pl.seed_everything(cfg.seed, workers=True)

    # get data loaders
    train_loader, train_dst, val_loader, _ = dataset.get_dataset(cfg)
    label_names = train_dst.label_names

    # init model
    model = None
    if cfg.model.name == "joint":
        model = MVJointVAE(cfg)
    elif cfg.model.name == "mixedprior":
        model = MVMixedPriorVAE(cfg)
    elif cfg.model.name == "unimodal":
        model = MVunimodalVAE(cfg)
    assert model is not None
    model.assign_label_names(label_names)
    summary = ModelSummary(model, max_depth=2)
    print(summary)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    wandb_logger = WandbLogger(
        name=cfg.log.wandb_run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        project=cfg.log.wandb_project_name,
        group=cfg.log.wandb_group,
        offline=cfg.log.wandb_offline,
        entity=cfg.log.wandb_entity,
        save_dir=cfg.log.dir_logs,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        monitor=cfg.checkpoint_metric,
        mode="min",
        save_last=True,
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

    if cfg.log.debug:
        trainer.logger.watch(model, log="all")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model.logger.log_metrics({f"final_scores/rec_loss": model.final_scores_rec_loss})
    model.logger.log_metrics(
        {f"final_scores/cond_rec_loss": model.final_scores_cond_rec_loss}
    )
    for m, key in enumerate(model.modality_names):
        model.logger.log_metrics(
            {
                f"final_scores/downstream_lr/aggregated/{key}": model.final_scores_lr_aggregated[
                    m
                ]
            }
        )
        model.logger.log_metrics(
            {
                f"final_scores/downstream_lr/unimodal/{key}": model.final_scores_lr_unimodal[
                    m
                ]
            }
        )
        if cfg.dataset.name == "celeba":
            for k, l_name in enumerate(label_names):
                model.logger.log_metrics(
                    {
                        f"final_scores/downstream_lr/aggregated/{key}/{l_name}": model.final_scores_lr_aggregated_alllabels[
                            m, k
                        ]
                    }
                )
                model.logger.log_metrics(
                    {
                        f"final_scores/downstream_lr/unimodal/{key}/{l_name}": model.final_scores_lr_unimodal_alllabels[
                            m, k
                        ]
                    }
                )

    for m, key in enumerate(model.modality_names):
        for m_tilde, key_tilde in enumerate(model.modality_names):
            model.logger.log_metrics(
                {
                    f"final_scores/coherence/{key}_to_{key_tilde}": model.final_scores_coh[
                        m, m_tilde, :
                    ].mean()
                }
            )
    for m, key in enumerate(model.modality_names):
        model.logger.log_metrics(
            {f"final_scores/likelihood/" + key: model.final_scores_nll[m]}
        )


if __name__ == "__main__":
    run_experiment()
