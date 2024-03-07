import os
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from sklearn.metrics import accuracy_score

from networks.ClfOdorRats import ClfOdorRats


class ClfRats(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.original_dims = [920, 790, 1040, 490, 460]
        self.clfs = nn.ModuleList(
            [
                ClfOdorRats(self.original_dims[m]).to(cfg.model.device)
                for m in range(cfg.dataset.num_views)
            ]
        )
        self.cfg = cfg
        self.loss = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.training_step_outputs = []

        # buffer for final scores
        # self.register_buffer("final_accuracy", torch.zeros(1))

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss, mean_acc = self.compute_loss("train", batch, out)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss, mean_acc = self.compute_loss("val", batch, out)
        self.validation_step_outputs.append(mean_acc)
        return loss

    def on_validation_epoch_end(self):
        mean_acc = torch.tensor(self.validation_step_outputs).mean()
        # self.final_accuracy = mean_acc
        self.validation_step_outputs.clear()  # free memory

    def compute_loss(self, str_set, batch, out):
        imgs, labels = batch
        preds, losses = out
        loss = losses.mean(dim=1).mean(dim=0)

        accs = []
        for m in range(self.cfg.dataset.num_views):
            loss_m = losses[:, m, :].mean(dim=0)
            pred_m = preds[:, m, :]
            acc_m = accuracy_score(labels.cpu(), np.argmax(pred_m.detach().cpu().numpy(), axis=1).astype(int))
            accs.append(acc_m)
            self.log(str_set + "/loss/v" + str(m), loss_m)
            self.log(str_set + "/accuracy/v" + str(m), acc_m)
        mean_acc = torch.tensor(accs).mean()
        self.log(str_set + "/loss/loss", loss)
        self.log(str_set + "/loss/mean_acc", mean_acc)
        return loss, mean_acc



    def forward(self, batch):
        imgs, labels = batch
        preds = torch.zeros((self.cfg.model.batch_size, self.cfg.dataset.num_views, 4), device=self.cfg.model.device)
        losses = torch.zeros((self.cfg.model.batch_size, self.cfg.dataset.num_views, 1), device=self.cfg.model.device)
        for m in range(self.cfg.dataset.num_views):
            imgs_m = imgs["m" + str(m)]
            pred_m = self.clfs[m](imgs_m)
            preds[:, m, :] = pred_m
            loss_m = self.loss(pred_m, labels)
            losses[:, m, :] = loss_m
        return preds, losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.model.lr,
        )
        return {
            "optimizer": optimizer,
        }

