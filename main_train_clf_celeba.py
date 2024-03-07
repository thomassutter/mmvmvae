import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

from utils import dataset
from clfs.celeba_clf import ClfCelebA

from config.MyClfConfig import MyClfConfig
from config.MyClfConfig import ModelConfig
from config.MyClfConfig import LogConfig
from config.MyClfConfig import CelebADataConfig

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(group="log", name="log", node=LogConfig)
cs.store(group="model", name="model", node=ModelConfig)
cs.store(group="dataset", name="CelebA", node=CelebADataConfig)
cs.store(name="base_config", node=MyClfConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config_clf")
def run_experiment(cfg: MyClfConfig):
    print(cfg)
    pl.seed_everything(cfg.seed, workers=True)

    # get data loaders
    train_loader, train_dst, val_loader, val_dst = dataset.get_dataset(cfg)

    # load model
    model = ClfCelebA(cfg)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.dataset.dir_clf,
        monitor=cfg.checkpoint_metric,
        mode="max",
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


if __name__ == "__main__":
    run_experiment()
