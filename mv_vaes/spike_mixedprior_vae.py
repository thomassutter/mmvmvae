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


class SPIKEMixedPriorVAE(SPIKEVAE):
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

        dists_out = []
        imgs_rec = {}

        for m in range(0, self.cfg.dataset.num_views):
            # encode views: img_m -> z_m
            img_m = images["m" + str(m)]
            mu_m, lv_m = self.encoders[m](img_m)
            z_m = self.reparametrize(mu_m, lv_m)

            # decode views: z_m -> img_hat_m
            img_hat_m = self.decoders[m](z_m)

            imgs_rec["m" + str(m)] = img_hat_m

            dist_out_m = [mu_m, lv_m]
            dists_out.append(dist_out_m)
        return (imgs_rec, dists_out)

    def compute_loss(self, str_set, batch, forward_out):
        imgs, labels = batch
        imgs_rec = forward_out[0]
        dists_out = forward_out[1]

        # kl divergence of latent distribution
        klds = []
        for m in range(self.cfg.dataset.num_views):
            dist_m = dists_out[m]
            for m_tilde in range(self.cfg.dataset.num_views):
                dist_m_tilde = dists_out[m_tilde]
                kld_m_m_tilde = self.kl_div_z_two_dists(dist_m, dist_m_tilde)
                # KL(q_m | q_m_tilde) * (1-alpha)
                klds.append(kld_m_m_tilde.unsqueeze(1) * (1.0-self.cfg.model.stdnormweight))
            # add N(0,1) as a component
            kld_m = self.kl_div_z(dist_m)
            # KL(q_m | N(0,1)) * alpha * M
            klds.append(kld_m.unsqueeze(1) * self.cfg.model.stdnormweight * self.cfg.dataset.num_views)
        # SUM_{m}:( alpha * KL(q_m|N(0,1)) + (1-alpha)/M * SUM_{m_tilde}:KL(q_m|q_m_tilde) )
        # when alpha = 0: mixedprior
        # when alpha = 1: unimodal
        # when alpha = 1/(M+1): mixedpriorstdnorm
        klds_sum = torch.cat(klds, dim=1).sum(dim=1) / self.cfg.dataset.num_views

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
