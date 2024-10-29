import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from networks.NetworkTextClfCUB import ClfText
from networks.NetworkImgClfCUB import ClfImg

class ClfCUB(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.clfs = nn.ModuleList(
            [
                ClfText().to(cfg.model.device),
                ClfImg().to(cfg.model.device)
            ]
        )

        self.cfg = cfg
        self.loss = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.training_step_outputs = []

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        out = self.forward(self.cfg, batch)
        imgs, labels = batch
        preds, losses = out
        loss = losses.mean(dim=1).mean(dim=0)

        self.log("train/loss/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(self.cfg, batch)
        imgs, labels = batch
        preds, losses = out
        loss = losses.mean(dim=1).mean(dim=0)
        self.validation_step_outputs.append([preds, labels])
        for m in range(self.cfg.dataset.num_views):
            loss_m = losses[:, m, :].mean(dim=0)
            self.log("val/loss/v" + str(m), loss_m)

        return loss

    def on_validation_epoch_end(self):
        preds = []
        labels = []
        for i in range(len(self.validation_step_outputs)):
            preds_i, labels_i = self.validation_step_outputs[i]
            preds.append(preds_i)
            labels.append(labels_i)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        self.validation_step_outputs.clear()

        n_labels = labels.shape[1]
        aps = []
        aurocs = []
        for m in range(self.cfg.dataset.num_views):
            pred_m = preds[:, m, :]
            aps_m = torch.zeros(n_labels)
            aurocs_m = torch.zeros(n_labels)
            for k, str_l in enumerate(self.cfg.dataset.label_names):
                aurocs_m[k] = roc_auc_score(
                    labels[:, k].cpu(), pred_m[:, k].detach().cpu().numpy()
                ) 
                self.log("val/auroc/v" + str(m) + "/" + str_l, aurocs_m[k])
                aps_m[k] = average_precision_score(
                    labels[:, k].cpu(), pred_m[:, k].detach().cpu().numpy()
                ) 
                self.log("val/ap/v" + str(m) + "/" + str_l, aps_m[k])
            aps.append(aps_m.mean())
            aurocs.append(aurocs_m.mean())
            self.log("val/ap/v" + str(m) + "_mean", aps_m.mean())
            self.log("val/auroc/v" + str(m) + "_mean", aurocs_m.mean())
        mean_ap = torch.tensor(aps).mean()
        mean_auroc = torch.tensor(aurocs).mean()
        self.log("val/auroc/mean", mean_auroc)
        self.log("val/ap/mean", mean_ap)
        self.log("val/loss/mean_metric", mean_auroc)
        self.validation_step_outputs.clear()  # free memory

    def forward(self, cfg, batch):
        data, labels = batch
        n_labels = labels.shape[1]
        preds = torch.zeros(
            (cfg.model.batch_size_eval, cfg.dataset.num_views, n_labels),
            device=cfg.model.device,
        )
        losses = torch.zeros(
            (cfg.model.batch_size_eval, cfg.dataset.num_views, 1), device=cfg.model.device
        )
        # for m, m_key in enumerate(data.keys()):
        #     m_val = data[m_key]
        #     print(m, m_key, m_val.shape)
            
        for m, m_key in enumerate(data.keys()):
            m_val = data[m_key]
            pred_m = self.clfs[m](m_val)
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
