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
import matplotlib.pyplot as plt
from utils.eval import train_clf_lr, eval_clf_lr
from utils.eval import conditional_generation
from utils.eval import calc_coherence
from utils.eval import load_modality_clfs_rats

class SPIKEVAE(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.original_dims = [920, 790, 1040, 490, 460]
        # self.original_dims = [790, 490, 460]
        # self.original_dims = [490, 460]
        self.encoders = nn.ModuleList(
            [
                Encoder(cfg.model.latent_dim, self.original_dims[m]).to(cfg.model.device)
                for m in range(cfg.dataset.num_views)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                Decoder(cfg.model.latent_dim, self.original_dims[m]).to(cfg.model.device)
                for m in range(cfg.dataset.num_views)
            ]
        )

        self.validation_step_outputs = []
        self.training_step_outputs = []

        self.save_hyperparameters()
        self.register_buffer("final_accuracies_lr", torch.zeros(cfg.dataset.num_views))
        self.register_buffer("final_accuracies_lr_one_clf", torch.zeros(cfg.dataset.num_views))
        self.register_buffer("final_rec_loss_coh", torch.zeros(cfg.dataset.num_views*cfg.dataset.num_views))
        self.register_buffer("final_accuracies_coh", torch.zeros(cfg.dataset.num_views*cfg.dataset.num_views))
        if self.cfg.model.name == "joint":
            self.register_buffer("final_accuracies_lr_orig", torch.zeros(cfg.dataset.num_views))
            self.register_buffer("final_accuracies_lr_one_clf_orig", torch.zeros(cfg.dataset.num_views))

    def training_step(self, batch, batch_idx):
        out = self.forward(batch, resample=True)
        loss = self.compute_loss("train", batch, out[:2])
        bs = self.cfg.model.batch_size
        if len(self.training_step_outputs) * bs < self.cfg.eval.num_samples_train:
            self.training_step_outputs.append([out, batch])
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch, resample=self.cfg.model.resample_eval)
        loss = self.compute_loss("val", batch, out[:2])

        if self.cfg.eval.coherence:
            rec_loss, acc_coh = self.evaluate_conditional_generation(out, batch)
        else:
            rec_loss, acc_coh = None, None
        self.validation_step_outputs.append([out, batch, rec_loss, acc_coh])
        self.log_additional_values(out)
        return loss

    def evaluate_conditional_generation(self, out, batch):
        dists_out = out[-1]
        imgs = batch[0]
        labels = batch[1]
        n_views = self.cfg.dataset.num_views
        clfs_coherence = load_modality_clfs_rats(self.cfg)

        rec_loss = torch.zeros((1, n_views*n_views))
        accs = torch.zeros((1, n_views*n_views))

        for m in range(n_views):
            mu_m, lv_m = dists_out[m]
            imgs_m_gen = {}
            for m_tilde in range(n_views): 
                z_m = self.reparametrize(mu_m, lv_m)
                img_c_gen_m_tilde = self.decoders[m_tilde](z_m)[0]
                imgs_m_gen["m" + str(m_tilde)] = img_c_gen_m_tilde
            # reconstruction loss
            rec_loss_m = self.compute_rec_loss_view_to_view(imgs, imgs_m_gen) # (128, 5), (128, 5)
            rec_loss[0, m*n_views:(m+1)*n_views] = rec_loss_m
            # accuracy of the conditional generated rat
            accs_m = calc_coherence(clfs_coherence, imgs_m_gen, labels)
            accs[0, m*n_views:(m+1)*n_views ] = accs_m
        return rec_loss, accs

    def on_validation_epoch_end(self):

        enc_mu_train = {str(m): [] for m in range(self.cfg.dataset.num_views)}
        labels_train = []
        n_hat_train = []
        assignments_train = []
        if len(self.training_step_outputs) == 0:
            return
        # select samples for training of classifier
        for idx, train_out in enumerate(self.training_step_outputs):
            out, batch = train_out
            imgs, labels = batch
            dists_out = out[1]
            for m in range(self.cfg.dataset.num_views):
                mu_m, lv_m = dists_out[m]
                enc_mus_m = enc_mu_train[str(m)]
                enc_mus_m.append(mu_m)
                enc_mu_train[str(m)] = enc_mus_m
            labels_train.append(labels)
        for m in range(self.cfg.dataset.num_views):
            enc_mu_m_train = enc_mu_train[str(m)]
            enc_mu_m_train = torch.cat(enc_mu_m_train, dim=0)
            enc_mu_train[str(m)] = enc_mu_m_train
        labels_train = torch.cat(labels_train, dim=0)
        # do everything using training output before this line
        self.training_step_outputs.clear()  # free memory

        clfs = []
        if self.cfg.eval.eval_downstream_task:
            for m in range(self.cfg.dataset.num_views):
                enc_mu_m_train = enc_mu_train[str(m)]
                clf_m = train_clf_lr(
                    enc_mu_m_train,
                    labels_train,
                )
                clfs.append(clf_m)
            # the shared clf for all views
            enc_mu_all_train = torch.cat(list(enc_mu_train.values()), dim=0)
            labels_all_train = labels_train.repeat(self.cfg.dataset.num_views)
            clf_one_clf = train_clf_lr(
                enc_mu_all_train,
                labels_all_train,
            )

        enc_mu_val = {str(m): [] for m in range(self.cfg.dataset.num_views)}
        enc_lv_val = {str(m): [] for m in range(self.cfg.dataset.num_views)}
        labels_val = []
        rec_loss_val = []
        acc_coherence = []
        enc_mu_val_orig = {str(m): [] for m in range(self.cfg.dataset.num_views)}

        for idx, val_out in enumerate(self.validation_step_outputs):
            out, batch, rec_loss, acc_coh = val_out 
            imgs, labels = batch
            dists_out = out[1]
            dists_out_orig = out[-1]
            for m in range(self.cfg.dataset.num_views):
                mu_m, lv_m = dists_out[m]
                enc_mus_m = enc_mu_val[str(m)]
                enc_lvs_m = enc_lv_val[str(m)]
                enc_mus_m.append(mu_m)
                enc_lvs_m.append(lv_m)
                enc_mu_val[str(m)] = enc_mus_m
                enc_lv_val[str(m)] = enc_lvs_m

                mu_m_orig, _ = dists_out_orig[m]
                enc_mus_m_orig = enc_mu_val_orig[str(m)]
                enc_mus_m_orig.append(mu_m_orig)
                enc_mu_val_orig[str(m)] = enc_mus_m_orig

            labels_val.append(labels)
            rec_loss_val.append(rec_loss)
            acc_coherence.append(acc_coh)
        self.log_additional_values_val()
        self.validation_step_outputs.clear()  # free memory

        for m in range(self.cfg.dataset.num_views):
            enc_mu_m_val = enc_mu_val[str(m)]
            enc_mu_m_val = torch.cat(enc_mu_m_val, dim=0)
            enc_mu_val[str(m)] = enc_mu_m_val
            enc_lv_m_val = enc_lv_val[str(m)]
            enc_lv_m_val = torch.cat(enc_lv_m_val, dim=0)
            enc_lv_val[str(m)] = enc_lv_m_val

            enc_mu_m_val_orig = enc_mu_val_orig[str(m)]
            enc_mu_m_val_orig = torch.cat(enc_mu_m_val_orig, dim=0)
            enc_mu_val_orig[str(m)] = enc_mu_m_val_orig
        labels_val = torch.cat(labels_val, dim=0)

        if self.cfg.eval.eval_downstream_task:
            # downstream task - latent visualization
            colors = ['deepskyblue', 'tan', 'mediumseagreen', 'purple', 'CORAL']
            # marker_list = [".", "_", "|", "x", "3"]
            if self.cfg.model.latent_dim == 2:
                # all the views/rats
                plt.figure(figsize=(15, 9))
                plt.clf()
                for m in range(self.cfg.dataset.num_views):
                    if False:
                        enc_mu_m_val = enc_mu_val_orig[str(m)]
                        plt.scatter(enc_mu_m_val.cpu().numpy()[:, 0],
                                    enc_mu_m_val.cpu().numpy()[:, 1],
                                    s=30,
                                    # marker=marker_list[m],
                                    marker='.',
                                    c=[colors[odor] for odor in labels_val.cpu().numpy().astype(int)])

                    enc_mu_m_val = enc_mu_val[str(m)]
                    plt.scatter(enc_mu_m_val.cpu().numpy()[:, 0],
                                enc_mu_m_val.cpu().numpy()[:, 1],
                                s=30,
                                # marker=marker_list[m],
                                marker='.',
                                c=[colors[odor] for odor in labels_val.cpu().numpy().astype(int)])
                plot_title = "scatter_spike_rats_"+"beta"+str(self.cfg.model.beta)+"stdnormweight"+str(self.cfg.model.stdnormweight)
                plt.title(plot_title)
                plot_name = plot_title+".png"
                plt.savefig(self.cfg.log.dir_logs+"/figs/"+plot_name)
                self.logger.log_image(key="scatter rats", images=[self.cfg.log.dir_logs+"/figs/"+plot_name])
                plt.close()

                # each view/rat
                plt.clf()
                for m in range(self.cfg.dataset.num_views):
                    enc_mu_m_val = enc_mu_val_orig[str(m)]
                    plt.figure(figsize=(15, 9))
                    plt.clf()
                    plt.scatter(enc_mu_m_val.cpu().numpy()[:, 0],
                                enc_mu_m_val.cpu().numpy()[:, 1],
                                s=30,
                                # marker=marker_list[m],
                                marker='.',
                                c=[colors[odor] for odor in labels_val.cpu().numpy().astype(int)])
                    plot_title = "scatter_spike_rat_"+str(m)+"beta"+str(self.cfg.model.beta)+"stdnormweight"+str(self.cfg.model.stdnormweight)
                    plt.title(plot_title)
                    plot_name = plot_title+".png"
                    plt.savefig(self.cfg.log.dir_logs+"/figs/"+plot_name)
                    self.logger.log_image(key="scatter rat " + str(m), images=[self.cfg.log.dir_logs+"/figs/"+plot_name])
                plt.close()

            if self.cfg.model.latent_dim == 3:
                # all the views/rats
                plt.clf()
                plt.subplot(projection='3d')
                for m in range(self.cfg.dataset.num_views):
                    enc_mu_m_val = enc_mu_val[str(m)]
                    plt.scatter(enc_mu_m_val.cpu().numpy()[:, 0],
                                enc_mu_m_val.cpu().numpy()[:, 1],
                                enc_mu_m_val.cpu().numpy()[:, 2],
                                marker='o',
                                c=[colors[odor] for odor in labels_val.cpu().numpy().astype(int)])
                plot_title = "3d_scatter_spike_rats_"+"beta"+str(self.cfg.model.beta)+"stdnormweight"+str(self.cfg.model.stdnormweight)
                plt.title(plot_title)
                plot_name = plot_title+".png"
                plt.savefig(self.cfg.log.dir_logs+"/figs/"+plot_name)
                self.logger.log_image(key="3d scatter rats", images=[self.cfg.log.dir_logs+"/figs/"+plot_name])
                plt.close()

                # each view/rat
                plt.clf()
                for m in range(self.cfg.dataset.num_views):
                    enc_mu_m_val = enc_mu_val[str(m)]
                    plt.clf()
                    plt.subplot(projection='3d')
                    plt.scatter(enc_mu_m_val.cpu().numpy()[:, 0],
                                enc_mu_m_val.cpu().numpy()[:, 1],
                                enc_mu_m_val.cpu().numpy()[:, 2],
                                marker='o',
                                c=[colors[odor] for odor in labels_val.cpu().numpy().astype(int)])
                    plot_title = "3d_scatter_spike_rat_"+str(m)+"beta"+str(self.cfg.model.beta)+"stdnormweight"+str(self.cfg.model.stdnormweight)
                    plt.title(plot_title)
                    plot_name = plot_title+".png"
                    plt.savefig(self.cfg.log.dir_logs+"/figs/"+plot_name)
                    self.logger.log_image(key="3d scatter rat " + str(m), images=[self.cfg.log.dir_logs+"/figs/"+plot_name])
                plt.close()

            # downstream task - logistic regression on latent space
            mean_acc_views = []
            mean_acc_views_one_clf = []
            for m in range(self.cfg.dataset.num_views):
                clf_m = clfs[m]
                enc_mu_m_val = enc_mu_val[str(m)]
                bal_acc_m = eval_clf_lr(
                    clf_m,
                    enc_mu_m_val,
                    labels_val,
                )
                self.log("val/downstream/acc_v" + str(m), bal_acc_m.mean())
                mean_acc_views.append(bal_acc_m.mean())
                # the shared clf for all views
                bal_acc_m_one_clf = eval_clf_lr(
                    clf_one_clf,
                    enc_mu_m_val,
                    labels_val,
                )
                self.log("val/downstream/acc_one_clf_v" + str(m), bal_acc_m_one_clf.mean())
                mean_acc_views_one_clf.append(bal_acc_m_one_clf.mean())
            # Save current final scores
            self.final_accuracies_lr = torch.tensor(mean_acc_views)
            self.final_accuracies_lr_one_clf = torch.tensor(mean_acc_views_one_clf)
            # logistic regression on latent space for joint model before aggregation
            if self.cfg.model.name == "joint":
                mean_acc_views_orig = []
                mean_acc_views_one_clf_orig = []
                for m in range(self.cfg.dataset.num_views):
                    clf_m = clfs[m]
                    enc_mu_m_val = enc_mu_val_orig[str(m)]
                    bal_acc_m = eval_clf_lr(
                        clf_m,
                        enc_mu_m_val,
                        labels_val,
                    )
                    self.log("val/downstream/orig_acc_v" + str(m), bal_acc_m.mean())
                    mean_acc_views_orig.append(bal_acc_m.mean())
                    # the shared clf for all views
                    bal_acc_m_one_clf = eval_clf_lr(
                        clf_one_clf,
                        enc_mu_m_val,
                        labels_val,
                    )
                    self.log("val/downstream/orig_acc_one_clf_v" + str(m), bal_acc_m_one_clf.mean())
                    mean_acc_views_one_clf_orig.append(bal_acc_m_one_clf.mean())
                # Save current final scores
                self.final_accuracies_lr_orig = torch.tensor(mean_acc_views_orig)
                self.final_accuracies_lr_one_clf_orig = torch.tensor(mean_acc_views_one_clf_orig)

        if self.cfg.eval.coherence:
            # coherence of conditional generation
            rec_loss_val = torch.cat(rec_loss_val)
            rec_loss_val_list = []
            acc_coherence = torch.cat(acc_coherence)
            accs_coh = []
            for m in range(self.cfg.dataset.num_views):
                for m_tilde in range(self.cfg.dataset.num_views):
                    rec_loss_m_m_tilde = rec_loss_val[:, m*self.cfg.dataset.num_views + m_tilde].mean()
                    self.log("val/coherence/rec_loss_v" + str(m) + "_to_" + str(m_tilde), rec_loss_m_m_tilde)
                    rec_loss_val_list.append(rec_loss_m_m_tilde)
                    acc_m_m_tilde = acc_coherence[:, m*self.cfg.dataset.num_views + m_tilde].mean()
                    self.log("val/coherence/acc_v" + str(m) + "_to_" + str(m_tilde), acc_m_m_tilde)
                    accs_coh.append(acc_m_m_tilde)
            self.final_rec_loss_coh = torch.tensor(rec_loss_val_list)
            self.final_accuracies_coh = torch.tensor(accs_coh)

    def kl_div_z(self, dist):
        mu, lv = dist
        prior_mu = torch.zeros_like(mu)
        prior_lv = torch.zeros_like(lv)
        prior_d = torch.distributions.normal.Normal(prior_mu, prior_lv.exp() + 1e-6)
        d1 = torch.distributions.normal.Normal(mu, lv.exp() + 1e-6)
        kld = torch.distributions.kl.kl_divergence(d1, prior_d).sum(dim=-1)
        return kld

    def kl_div_z_two_dists(self, dist1, dist2):
        mu1, lv1 = dist1
        mu2, lv2 = dist2
        d1 = torch.distributions.normal.Normal(mu1, lv1.exp() + 1e-6)
        d2 = torch.distributions.normal.Normal(mu2, lv2.exp() + 1e-6)
        kld = torch.distributions.kl.kl_divergence(d1, d2).sum(dim=-1)
        return kld

    def compute_rec_loss(self, imgs, imgs_rec):
        rec_loss_all = []
        for m in range(self.cfg.dataset.num_views):
            img_gt_m = imgs["m" + str(m)]
            img_rec_m = imgs_rec["m" + str(m)]
            # MSE loss
            mse = nn.MSELoss(reduction='none')
            rec_loss_m = mse(input=img_rec_m[0], target=img_gt_m).sum(-1)
            rec_loss_all.append(rec_loss_m.unsqueeze(1))
        rec_loss_avg = torch.cat(rec_loss_all, dim=1).sum(dim=1)
        return rec_loss_avg

    def compute_rec_loss_each_view(self, imgs, imgs_rec):
        rec_loss_all = []
        for m in range(self.cfg.dataset.num_views):
            img_gt_m = imgs["m" + str(m)]
            img_rec_m = imgs_rec["m" + str(m)]
            # MSE loss
            mse = nn.MSELoss(reduction='none')
            rec_loss_m = mse(input=img_rec_m[0], target=img_gt_m).sum(-1)
            rec_loss_all.append(rec_loss_m.unsqueeze(1))
        rec_loss_each_view = torch.cat(rec_loss_all, dim=1)
        return rec_loss_each_view

    def compute_rec_loss_view_to_view(self, imgs, imgs_rec):
        rec_loss_all = []
        for m in range(self.cfg.dataset.num_views):
            img_gt_m = imgs["m" + str(m)]
            img_rec_m = imgs_rec["m" + str(m)]
            # MSE loss
            mse = nn.MSELoss(reduction='none')
            rec_loss_m = mse(input=img_rec_m, target=img_gt_m).sum(-1)
            rec_loss_all.append(rec_loss_m.unsqueeze(1))
        rec_loss_each_view = torch.cat(rec_loss_all, dim=1).mean(dim=0)
        return rec_loss_each_view

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.model.lr,
        )
        return {
            "optimizer": optimizer,
        }

    def aggregate_latents(self, mus, lvs):
        batch_size, num_views, num_latents = mus.shape
        mu_agg = (mus.sum(dim=1) / float(num_views)).squeeze(1)
        lv_agg = (lvs.exp().sum(dim=1) / float(num_views)).log().squeeze(1)
        return mu_agg, lv_agg

    def reparametrize(self, mu, log_sigma):
        """
        Reparametrized sampling from gaussian
        """
        dist = torch.distributions.normal.Normal(mu, log_sigma.exp() + 1e-6)
        return dist.rsample()
