import os
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

from torchvision.utils import make_grid
import torchvision.transforms.functional as F

from networks.NetworksRatsspike import Encoder, Decoder

# from drpm.mv_drpm import MVDRPM
# from drpm.pl import PL

from mv_vaes.spike_vae import SPIKEVAE


class SPIKEJointVAE(SPIKEVAE):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_hyperparameters()

    def log_additional_values(self, out):
        pass

    def log_additional_values_val(self):
        pass

    def forward(self, batch, resample):
        images = batch[0]
        labels = batch[1]
        dists_out_orig = []

        mus = []
        lvs = []
        # encode views
        for m in range(0, self.cfg.dataset.num_views):
            img_m = images["m" + str(m)]
            mu_m, lv_m = self.encoders[m](img_m)
            dists_out_orig.append([mu_m, lv_m])
            mus.append(mu_m.unsqueeze(1))
            lvs.append(lv_m.unsqueeze(1))
        mus = torch.cat(mus, dim=1)
        lvs = torch.cat(lvs, dim=1)

        mu_out, lv_out = self.aggregate_latents(mus, lvs)
        z_out = self.reparametrize(mu_out, lv_out)

        # decode views
        dists_out = []
        imgs_rec = {}
        for m in range(0, self.cfg.dataset.num_views):
            img_hat_m = self.decoders[m](z_out)
            imgs_rec["m" + str(m)] = img_hat_m

            dist_out_m = [mu_out, lv_out]
            dists_out.append(dist_out_m)
        return (imgs_rec, dists_out, dists_out_orig)

    def compute_loss(self, str_set, batch, forward_out):
        imgs, labels = batch
        imgs_rec = forward_out[0]
        dists_out = forward_out[1]

        # kl divergence of latent distribution
        klds = []
        for m in range(self.cfg.dataset.num_views):
            dist_m = dists_out[m]
            kld_m = self.kl_div_z(dist_m)
            klds.append(kld_m.unsqueeze(1))
        klds_sum = torch.cat(klds, dim=1).sum(dim=1)

        ## compute reconstruction loss/ conditional log-likelihood out data
        ## given latents
        loss_rec = self.compute_rec_loss(imgs, imgs_rec)
        loss_rec_all_views = self.compute_rec_loss_each_view(imgs, imgs_rec).mean(dim=0)

        beta = self.cfg.model.beta
        loss_mv_vae = (loss_rec + beta * klds_sum).mean(dim=0)
        total_loss = loss_mv_vae
        # logging
        self.log(str_set + "/loss/klds_avg", klds_sum.mean(dim=0))
        self.log(str_set + "/loss/loss_rec", loss_rec.mean(dim=0))
        self.log(str_set + "/loss/mv_vae", loss_mv_vae)
        self.log(str_set + "/loss/loss", total_loss)
        
        for m in range(self.cfg.dataset.num_views):
            self.log(str_set + "/loss/loss_rec_v"+str(m), loss_rec_all_views[m])

        return total_loss
