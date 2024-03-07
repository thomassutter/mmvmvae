import sys
import math

import torch
from torch import nn
import pytorch_lightning as pl
import wandb

from torchvision.utils import make_grid
import torchvision.transforms.functional as t_f

from networks.NetworksImgCelebA import EncoderImg, DecoderImg
from networks.NetworksTextCelebA import EncoderText, DecoderText
from networks.ConvNetworksPolyMNIST import Encoder, Decoder
from networks.ConvNetworksPolyMNIST import ResnetEncoder, ResnetDecoder
from utils.eval import train_clf_lr_PM, eval_clf_lr_PM
from utils.eval import train_clf_lr_celeba, eval_clf_lr_celeba
from utils.eval import generate_samples
from utils.eval import conditional_generation
from utils.eval import calc_coherence_acc, calc_coherence_ap
from utils.eval import load_modality_clfs
from utils.eval import from_preds_to_acc
from utils.eval import from_preds_to_ap

from utils.text import create_txt_image


class MVVAE(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.dataset.name.startswith("PM"):
            if not cfg.model.use_resnets:
                self.encoders = nn.ModuleList(
                    [
                        Encoder(cfg.model.latent_dim).to(cfg.model.device)
                        for _ in range(cfg.dataset.num_views)
                    ]
                )
                self.decoders = nn.ModuleList(
                    [
                        Decoder(cfg.model.latent_dim).to(cfg.model.device)
                        for _ in range(cfg.dataset.num_views)
                    ]
                )
            else:
                self.encoders = nn.ModuleList(
                    [
                        ResnetEncoder(cfg).to(cfg.model.device)
                        for _ in range(cfg.dataset.num_views)
                    ]
                )
                self.decoders = nn.ModuleList(
                    [
                        ResnetDecoder(cfg).to(cfg.model.device)
                        for _ in range(cfg.dataset.num_views)
                    ]
                )
        elif cfg.dataset.name.startswith("celeba"):
            self.encoders = nn.ModuleList(
                [
                    EncoderImg(cfg).to(cfg.model.device),
                    EncoderText(cfg).to(cfg.model.device),
                ]
            )
            self.decoders = nn.ModuleList(
                [
                    DecoderImg(cfg).to(cfg.model.device),
                    DecoderText(cfg).to(cfg.model.device),
                ]
            )

        if cfg.dataset.name.startswith("PM"):
            self.train_clf_lr = train_clf_lr_PM
            self.eval_clf_lr = eval_clf_lr_PM
            self.eval_downstream_task = self.eval_downstream_task_PM
            self.from_preds_to_clf_metric = from_preds_to_acc
            self.calc_coherence = calc_coherence_acc
            self.modality_names = [
                "m" + str(m) for m in range(0, cfg.dataset.num_views)
            ]
            self.ref_mod_d_size = 3 * 28 * 28
            self.modalities_size = {
                "m" + str(m): 3 * 28 * 28 for m in range(cfg.dataset.num_views)
            }
        elif cfg.dataset.name.startswith("celeba"):
            self.train_clf_lr = train_clf_lr_celeba
            self.eval_clf_lr = eval_clf_lr_celeba
            self.eval_downstream_task = self.eval_downstream_task_celeba
            self.calc_coherence = calc_coherence_ap
            self.from_preds_to_clf_metric = from_preds_to_ap
            self.modality_names = ["img", "text"]
            self.betas = {"img": cfg.dataset.beta_img, "text": cfg.dataset.beta_text}
            if cfg.dataset.use_rec_weight and cfg.dataset.include_channels_rec_weight:
                self.ref_mod_d_size = 3 * cfg.dataset.img_size * cfg.dataset.img_size
                self.modalities_size = {
                    "img": 3 * cfg.dataset.img_size * cfg.dataset.img_size,
                    "text": cfg.dataset.len_sequence,
                }
            elif cfg.dataset.use_rec_weight:
                self.ref_mod_d_size = cfg.dataset.img_size * cfg.dataset.img_size
                self.modalities_size = {
                    "img": cfg.dataset.img_size * cfg.dataset.img_size,
                    "text": cfg.dataset.len_sequence,
                }
            else:
                self.ref_mod_d_size = cfg.dataset.img_size * cfg.dataset.img_size
                self.modalities_size = {
                    "img": cfg.dataset.img_size * cfg.dataset.img_size,
                    "text": cfg.dataset.img_size * cfg.dataset.img_size,
                }

        if cfg.model.temp_annealing == "cosine":
            self.compute_current_temperature = self.cos_annealing
        elif cfg.model.temp_annealing == "linear":
            self.compute_current_temperature = self.linear_annealing
        elif cfg.model.temp_annealing == "exp":
            self.compute_current_temperature = self.exp_annealing
        else:
            print("annealing schedule not known...exit")
            sys.exit()

        self.validation_step_outputs = []
        self.training_step_outputs = []

        self.save_hyperparameters()

        # buffer for final scores
        self.register_buffer("final_scores_rec_loss", torch.zeros(1))
        self.register_buffer("final_scores_cond_rec_loss", torch.zeros(1))
        self.register_buffer(
            "final_scores_lr_unimodal", torch.zeros(cfg.dataset.num_views)
        )
        self.register_buffer(
            "final_scores_lr_aggregated", torch.zeros(cfg.dataset.num_views)
        )
        self.register_buffer(
            "final_scores_lr_unimodal_alllabels",
            torch.zeros(cfg.dataset.num_views, cfg.dataset.num_labels),
        )
        self.register_buffer(
            "final_scores_lr_aggregated_alllabels",
            torch.zeros(cfg.dataset.num_views, cfg.dataset.num_labels),
        )
        self.register_buffer(
            "final_scores_coh",
            torch.zeros(
                (cfg.dataset.num_views, cfg.dataset.num_views, cfg.dataset.num_labels)
            ),
        )

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss, _ = self.compute_loss("train", batch, out)
        bs = self.cfg.model.batch_size
        if len(self.training_step_outputs) * bs < self.cfg.eval.num_samples_train:
            self.training_step_outputs.append([out, batch])
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss, rec_loss = self.compute_loss("val", batch, out)

        if (self.current_epoch + 1) % self.cfg.log.coherence_logging_frequency == 0:
            if self.cfg.eval.coherence:
                pred_coh, cond_rec_loss = self.evaluate_conditional_generation(
                    out, batch
                )
            else:
                pred_coh, cond_rec_loss = None, None
        else:
            pred_coh, cond_rec_loss = None, None

        self.last_val_batch = batch
        self.validation_step_outputs.append(
            [out, batch[1], pred_coh, cond_rec_loss, rec_loss]
        )
        self.log_additional_values(out)
        return loss

    def forward(self, batch):
        pass

    def evaluate_conditional_generation(self, out, batch):
        dists_enc_out = out[2]
        labels = batch[1]
        data = batch[0]
        n_views = self.cfg.dataset.num_views
        clfs_coherence = load_modality_clfs(self.cfg)

        preds = torch.zeros(
            (
                self.cfg.model.batch_size,
                n_views,
                n_views,
                self.cfg.dataset.n_clfs_outputs,
            ),
            device=self.cfg.model.device,
        )
        # for m in range(n_views):
        cond_rec = {}
        for m, key in enumerate(self.modality_names):
            mu_m, lv_m = dists_enc_out[key]
            mods_m_gen = {}
            # for m_tilde in range(n_views):
            for m_tilde, key_tilde in enumerate(self.modality_names):
                z_m = self.reparametrize(mu_m, lv_m)
                mod_c_gen_m_tilde = self.decoders[m_tilde](z_m)
                mods_m_gen[key_tilde] = mod_c_gen_m_tilde[0]
                if m_tilde == m:
                    cond_rec[key] = mod_c_gen_m_tilde
            preds_m = self.calc_coherence(self.cfg, clfs_coherence, mods_m_gen, labels)
            preds[:, m] = preds_m
        cond_rec_loss, _, _ = self.compute_rec_loss(data, cond_rec)
        return preds, cond_rec_loss

    def on_validation_epoch_end(self):
        enc_mu_out_train = {key: [] for key in self.modality_names}
        enc_mu_enc_train = {key: [] for key in self.modality_names}
        labels_train = []
        if len(self.training_step_outputs) == 0:
            return
        # select samples for training of classifier
        for idx, train_out in enumerate(self.training_step_outputs):
            out, batch = train_out
            data, labels = batch
            dists_out = out[1]
            dists_enc = out[2]
            # for m in range(self.cfg.dataset.num_views):
            for m, key in enumerate(self.modality_names):
                mu_out_m, lv_out_m = dists_out[key]
                mu_enc_m, lv_enc_m = dists_enc[key]
                enc_mus_out_m = enc_mu_out_train[key]
                enc_mus_enc_m = enc_mu_enc_train[key]
                enc_mus_out_m.append(mu_out_m)
                enc_mus_enc_m.append(mu_enc_m)
                enc_mu_out_train[key] = enc_mus_out_m
                enc_mu_enc_train[key] = enc_mus_enc_m
            labels_train.append(labels)
        # for m in range(self.cfg.dataset.num_views):
        for m, key in enumerate(self.modality_names):
            enc_mu_out_m_train = enc_mu_out_train[key]
            enc_mu_out_m_train = torch.cat(enc_mu_out_m_train, dim=0)
            enc_mu_out_train[key] = enc_mu_out_m_train
            enc_mu_enc_m_train = enc_mu_enc_train[key]
            enc_mu_enc_m_train = torch.cat(enc_mu_enc_m_train, dim=0)
            enc_mu_enc_train[key] = enc_mu_enc_m_train
        labels_train = torch.cat(labels_train, dim=0)
        # do everything using training output before this line
        self.training_step_outputs.clear()  # free memory

        if (self.current_epoch + 1) % self.cfg.log.downstream_logging_frequency == 0:
            clfs_out = []
            clfs_enc = []
            if self.cfg.eval.eval_downstream_task:
                # train linear clfs on representations that are fed into decoder
                # i.e. after aggregation for joint models
                for m, key in enumerate(data.keys()):
                    enc_mu_out_m_train = enc_mu_out_train[key]
                    clf_out_m = self.train_clf_lr(
                        enc_mu_out_m_train,
                        labels_train,
                    )
                    clfs_out.append(clf_out_m)

                # train linear clfs on representations right after encoders
                # i.e. before aggregation for joint models
                for m, key in enumerate(data.keys()):
                    enc_mu_enc_m_train = enc_mu_enc_train[key]
                    clf_enc_m = self.train_clf_lr(
                        enc_mu_enc_m_train,
                        labels_train,
                    )
                    clfs_enc.append(clf_enc_m)

        enc_mu_out_val = {key: [] for key in self.modality_names}
        enc_lv_out_val = {key: [] for key in self.modality_names}
        enc_mu_enc_val = {key: [] for key in self.modality_names}
        enc_lv_enc_val = {key: [] for key in self.modality_names}
        labels_val = []
        preds_coherence = []
        cond_rec_loss = []
        rec_loss = []
        for idx, val_out in enumerate(self.validation_step_outputs):
            (
                out,
                labels,
                pred_coh,
                cond_rec_loss_batch,
                rec_loss_batch,
            ) = val_out
            # imgs, labels = batch
            dists_out = out[1]
            dists_enc_out = out[2]
            for m, key in enumerate(self.modality_names):
                mu_out_m, lv_out_m = dists_out[key]
                mu_enc_m, lv_enc_m = dists_enc_out[key]
                enc_mus_out_m = enc_mu_out_val[key]
                enc_lvs_out_m = enc_lv_out_val[key]
                enc_mus_enc_m = enc_mu_enc_val[key]
                enc_lvs_enc_m = enc_lv_enc_val[key]
                enc_mus_out_m.append(mu_out_m)
                enc_lvs_out_m.append(lv_out_m)
                enc_mus_enc_m.append(mu_enc_m)
                enc_lvs_enc_m.append(lv_enc_m)
                enc_mu_out_val[key] = enc_mus_out_m
                enc_lv_out_val[key] = enc_lvs_out_m
                enc_mu_enc_val[key] = enc_mus_enc_m
                enc_lv_enc_val[key] = enc_lvs_enc_m
            labels_val.append(labels)
            preds_coherence.append(pred_coh)
            cond_rec_loss.append(cond_rec_loss_batch)
            rec_loss.append(rec_loss_batch)
        self.log_additional_values_val()
        self.validation_step_outputs.clear()  # free memory

        self.log("val/loss/avg_rec_loss_epoch", torch.cat(rec_loss).mean())
        self.final_scores_rec_loss = torch.cat(rec_loss).mean()

        for m, key in enumerate(self.modality_names):
            enc_mu_out_m_val = enc_mu_out_val[key]
            enc_mu_out_m_val = torch.cat(enc_mu_out_m_val, dim=0)
            enc_mu_out_val[key] = enc_mu_out_m_val
            enc_lv_out_m_val = enc_lv_out_val[key]
            enc_lv_out_m_val = torch.cat(enc_lv_out_m_val, dim=0)
            enc_lv_out_val[key] = enc_lv_out_m_val
            enc_mu_enc_m_val = enc_mu_enc_val[key]
            enc_mu_enc_m_val = torch.cat(enc_mu_enc_m_val, dim=0)
            enc_mu_enc_val[key] = enc_mu_enc_m_val
            enc_lv_enc_m_val = enc_lv_enc_val[key]
            enc_lv_enc_m_val = torch.cat(enc_lv_enc_m_val, dim=0)
            enc_lv_enc_val[key] = enc_lv_enc_m_val
        labels_val = torch.cat(labels_val, dim=0)

        if (self.current_epoch + 1) % self.cfg.log.coherence_logging_frequency == 0:
            if self.cfg.eval.coherence:
                # coherence of conditional generation
                pred_coherence = torch.cat(preds_coherence)
                acc_coh = self.from_preds_to_clf_metric(
                    pred_coherence, labels_val, self.modality_names
                )
                self.final_scores_coh = acc_coh
                for m, key in enumerate(self.modality_names):
                    for m_tilde, key_tilde in enumerate(self.modality_names):
                        accs_m_m_tilde = acc_coh[m, m_tilde, :].mean()
                        self.log(
                            "val/coherence/" + key + "_to_" + key_tilde,
                            accs_m_m_tilde,
                        )
                if self.cfg.dataset.name == "celeba":
                    self.coherence_plot_all_labels_celeba(acc_coh)
                self.log(
                    "val/condition_generation/avg_rec_loss",
                    torch.cat(cond_rec_loss).mean(),
                )
                self.final_scores_cond_rec_loss = torch.cat(cond_rec_loss).mean()

        if (self.current_epoch + 1) % self.cfg.log.downstream_logging_frequency == 0:
            if self.cfg.eval.eval_downstream_task:
                scores_agg = self.eval_downstream_task(
                    "aggregated", clfs_out, enc_mu_out_val, labels_val
                )
                scores_unimodal = self.eval_downstream_task(
                    "unimodal", clfs_enc, enc_mu_enc_val, labels_val
                )
                # Save current final scores
                self.final_scores_lr_unimodal = scores_unimodal.mean(dim=1)
                self.final_scores_lr_aggregated = scores_agg.mean(dim=1)
                self.final_scores_lr_unimodal_alllabels = scores_unimodal
                self.final_scores_lr_aggregated_alllabels = scores_agg

        if (self.current_epoch + 1) % self.cfg.log.img_plotting_frequency == 0:
            n_samples_plot = min(100, self.cfg.model.batch_size)
            n_samples_row = int(math.sqrt(n_samples_plot))
            # plotting samples
            # generate samples
            z = torch.randn(
                (100, self.cfg.model.latent_dim), device=self.cfg.model.device
            )
            random_samples = generate_samples(self.decoders, z)
            # for m in range(self.cfg.dataset.num_views):
            for m, key in enumerate(self.modality_names):
                random_gen_m = random_samples[m][:n_samples_plot]
                if key == "text":
                    random_gen_m, random_txt_samples = create_txt_image(
                        self.cfg, random_gen_m
                    )
                    self.log_txt_samples(
                        random_txt_samples,
                        "val/txt_samples/random",
                        "random generations",
                    )
                imgs_grid_m = make_grid(random_gen_m, nrow=n_samples_row)
                imgs_grid_m = t_f.to_pil_image(imgs_grid_m)
                self.logger.log_image(
                    key="random generations " + key,
                    images=[wandb.Image(imgs_grid_m)],
                )

            # reconstructions
            # for m in range(self.cfg.dataset.num_views):
            for m, key in enumerate(self.modality_names):
                mod_m = self.last_val_batch[0][key][:n_samples_plot]
                mod_rec_m = out[0][key][0][:n_samples_plot]
                if key == "text":
                    mod_rec_m, txt_samples_rec = create_txt_image(self.cfg, mod_rec_m)
                    mod_m, txt_samples_gt = create_txt_image(self.cfg, mod_m)
                    self.log_txt_samples(
                        txt_samples_rec, "val/txt_samples", "reconstructions"
                    )
                    self.log_txt_samples(txt_samples_gt, "val/txt_samples", "gt")
                mod_grid_m = make_grid(
                    torch.cat([mod_m, mod_rec_m], dim=0), nrow=n_samples_row
                )
                mod_grid_m = t_f.to_pil_image(mod_grid_m)
                self.logger.log_image(
                    key="reconstructions " + key, images=[wandb.Image(mod_grid_m)]
                )

            # conditional generations
            # to start with: we only do conditional generation based on a single modality
            # and generate the remaining modalities
            # TODO: change to all possible conditional generration paths
            # for m in range(self.cfg.dataset.num_views):
            for m, key in enumerate(self.modality_names):
                mod_m = self.last_val_batch[0][key][-n_samples_plot:]
                mu_m_val = enc_mu_enc_val[key][-n_samples_plot:]
                lv_m_val = enc_lv_enc_val[key][-n_samples_plot:]
                dist_m = [mu_m_val, lv_m_val]
                mod_gen_m = conditional_generation(self, [dist_m])[0]
                if key == "text":
                    mod_m, _ = create_txt_image(self.cfg, mod_m)
                for m_tilde, key_tilde in enumerate(self.modality_names):
                    mod_gen_m_m_tilde = mod_gen_m[m_tilde]
                    if key_tilde == "text":
                        mod_gen_m_m_tilde, txt_m_m_tilde = create_txt_image(
                            self.cfg, mod_gen_m_m_tilde
                        )
                        mod_gen_m_m_tilde = mod_gen_m_m_tilde.to(self.cfg.model.device)
                        self.log_txt_samples(
                            txt_m_m_tilde,
                            "val/txt_samples",
                            f"cond_gen_{key}_{key_tilde}",
                        )
                    mod_grid_m_m_tilde = make_grid(
                        torch.cat(
                            [mod_m.to(self.cfg.model.device), mod_gen_m_m_tilde], dim=0
                        ),
                        nrow=n_samples_row,
                    )
                    mod_grid_m_m_tilde = t_f.to_pil_image(mod_grid_m_m_tilde)
                    self.logger.log_image(
                        key="cond_gen_" + key + "_to_" + key_tilde,
                        images=[wandb.Image(mod_grid_m_m_tilde)],
                    )

    def log_txt_samples(self, txt_samples, str_txt, str_title):
        sample_ids = range(0, len(txt_samples))
        data = [[label, val] for (label, val) in zip(sample_ids, txt_samples)]
        table = wandb.Table(data=data, columns=["id", "txt_sample"])
        wandb.log(
            {
                str_txt: wandb.plot.bar(
                    table,
                    "id",
                    "txt_sample",
                    title=str_title,
                )
            }
        )

    def coherence_plot_all_labels_celeba(self, scores):
        for m, key in enumerate(self.modality_names):
            for m_tilde, tilde_key in enumerate(self.modality_names):
                data = [
                    [label, val]
                    for (label, val) in zip(self.label_names, scores[m, m_tilde, :])
                ]
                table = wandb.Table(data=data, columns=["label", "AP"])
                wandb.log(
                    {
                        "val/coherence/all_labels_"
                        + key: wandb.plot.bar(
                            table,
                            "label",
                            "AP",
                            title="Coherence " + key + " to " + tilde_key,
                        )
                    }
                )
                for k, l_name in enumerate(self.label_names):
                    self.log(
                        f"val/coherence/{key}_{tilde_key}/{l_name}",
                        scores[m, m_tilde, k],
                    )

    def eval_downstream_task_PM(self, str_ds, clfs, enc_mu_val, labels_val):
        scores = torch.zeros((self.cfg.dataset.num_views, 1))
        for m, key in enumerate(self.modality_names):
            clf_m = clfs[m]
            enc_mu_m_val = enc_mu_val[key]
            bal_acc_m = self.eval_clf_lr(
                clf_m,
                enc_mu_m_val,
                labels_val,
            )
            self.log("val/downstream/" + str_ds + "/" + key, bal_acc_m.mean())
            scores[m, 0] = bal_acc_m.mean()
        return scores

    def eval_downstream_task_celeba(self, str_ds, clfs, enc_mu_val, labels_val):
        n_labels = labels_val.shape[1]
        scores = torch.zeros((self.cfg.dataset.num_views, n_labels))
        for m, key in enumerate(self.modality_names):
            clf_m = clfs[m]
            enc_mu_m_val = enc_mu_val[key]
            scores_m = self.eval_clf_lr(
                clf_m,
                enc_mu_m_val,
                labels_val,
            )
            scores[m, :] = scores_m
            self.log("val/downstream/" + str_ds + "/" + key, scores_m.mean())
            data = [[label, val] for (label, val) in zip(self.label_names, scores_m)]
            table = wandb.Table(data=data, columns=["label", "AP"])
            wandb.log(
                {
                    "val/downstream_lr/"
                    + str_ds
                    + "/all_labels_"
                    + key: wandb.plot.bar(
                        table,
                        "label",
                        "AP",
                        title=str_ds + " Latent Representation Evaluation " + key,
                    )
                }
            )
            for k, l_name in enumerate(self.label_names):
                self.log(f"val/downstream/{str_ds}/{key}/{l_name}", scores_m[k])
        return scores

    def kl_div_z(self, dist):
        mu, lv = dist
        kld = self.calc_kl_divergence(mu, lv)
        return kld

    def kl_div_z_two_dists(self, dist1, dist2):
        mu1, lv1 = dist1
        mu2, lv2 = dist2
        kld = self.calc_kl_divergence(mu1, lv1, mu2, lv2)
        return kld

    def calc_kl_divergence(self, mu0, logvar0, mu1=None, logvar1=None, norm_value=None):
        if mu1 is None or logvar1 is None:
            kld = -0.5 * torch.sum(1 - logvar0.exp() - mu0.pow(2) + logvar0, dim=-1)
        else:
            kld = -0.5 * (
                torch.sum(
                    1
                    - logvar0.exp() / logvar1.exp()
                    - (mu0 - mu1).pow(2) / logvar1.exp()
                    + logvar0
                    - logvar1,
                    dim=-1,
                )
            )
        if norm_value is not None:
            kld = kld / float(norm_value)
        return kld

    def compute_rec_loss(self, data, data_rec):
        rec_loss_all = []
        rec_loss_mods = {}
        rec_loss_mods_weighted = {}
        # output probability x_m
        for key in data.keys():
            mod_gt_m = data[key]
            mod_rec_m = data_rec[key]
            rec_weight_m = float(self.ref_mod_d_size / self.modalities_size[key])
            if key == "text":
                mod_d_out_m = torch.distributions.one_hot_categorical.OneHotCategorical(
                    logits=mod_rec_m[0], validate_args=False
                )
                log_p_mod_m = mod_d_out_m.log_prob(mod_gt_m).sum(dim=[1])
            else:
                mod_d_out_m = torch.distributions.laplace.Laplace(
                    mod_rec_m[0], torch.tensor(0.75).to(self.device)
                )
                log_p_mod_m = mod_d_out_m.log_prob(mod_gt_m).sum(dim=[1, 2, 3])
            rec_loss_mods[key] = log_p_mod_m.mean(dim=0)
            rec_loss_mods_weighted[key] = (rec_weight_m * log_p_mod_m).mean(dim=0)
            rec_loss_all.append(rec_weight_m * log_p_mod_m.unsqueeze(1))
        rec_loss_avg = -torch.cat(rec_loss_all, dim=1).sum(dim=1)
        return rec_loss_avg, rec_loss_mods, rec_loss_mods_weighted

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.model.lr,
        )
        return {
            "optimizer": optimizer,
        }

    def aggregate_latents_avg(self, mus, lvs):
        num_views = mus.shape[1]
        mu_agg = (mus.sum(dim=1) / float(num_views)).squeeze(1)
        lv_agg = (lvs.exp().sum(dim=1) / float(num_views)).log().squeeze(1)
        return mu_agg, lv_agg

    def aggregate_latents_moe(self, mus, lvs):
        num_components = mus.shape[1]
        num_samples = mus.shape[0]
        w_mods = float(1 / num_components)
        idx_start = []
        idx_end = []
        for k in range(0, num_components):
            if k == 0:
                i_start = 0
            else:
                i_start = int(idx_end[k - 1])
            if k == num_components - 1:
                i_end = num_samples
            else:
                i_end = i_start + int(num_samples * w_mods)
            idx_start.append(i_start)
            idx_end.append(i_end)
        idx_end[-1] = num_samples
        mu_sel = torch.cat(
            [mus[idx_start[k] : idx_end[k], k, :] for k in range(num_components)], dim=0
        )
        logvar_sel = torch.cat(
            [lvs[idx_start[k] : idx_end[k], k, :] for k in range(num_components)], dim=0
        )
        return mu_sel, logvar_sel

    def aggregate_latents_poe(self, mus, lvs):
        log_precisions = -lvs
        precisions = log_precisions.exp()
        joint_log_precision = torch.logsumexp(log_precisions, dim=1)
        joint_lv = -joint_log_precision
        joint_var = joint_lv.exp()
        joint_mu = precisions.mul(mus).sum(dim=1) * joint_var
        return joint_mu, joint_lv

    def aggregate_latents_mopoe(self, mus, lvs):
        pass

    def reparametrize(self, mu, logvar):
        """
        Reparametrized sampling from gaussian
        """
        # dist = torch.distributions.normal.Normal(mu, log_sigma.exp() + 1e-6)
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
        # return dist.rsample()

    def exp_annealing(self, init_temp=None, final_temp=None, num_steps_annealing=None):
        """
        Compute temperature based on current step
        -> exponential temperature annealing
        """
        if init_temp is None:
            init_temp = self.cfg.init_temp
        if final_temp is None:
            final_temp = self.cfg.model.final_temp
        if num_steps_annealing is None:
            num_steps_annealing = self.cfg.model.num_steps_annealing
        rate = (math.log(final_temp + 1e-10) - math.log(init_temp + 1e-10)) / float(
            num_steps_annealing
        )
        curr_temp = max(init_temp * math.exp(rate * self.global_step), final_temp)
        return curr_temp

    def cos_annealing(self, init_temp=None, final_temp=None, num_steps_annealing=None):
        """
        Compute temperature based on current step
        -> cosine temperature annealing
        """
        if init_temp is None:
            init_temp = self.cfg.init_temp
        if final_temp is None:
            final_temp = self.cfg.model.final_temp
        if num_steps_annealing is None:
            num_steps_annealing = self.cfg.model.num_steps_annealing
        curr_temp_cos = final_temp + 0.5 * (init_temp - final_temp) * (
            1
            + torch.cos(
                torch.tensor((self.global_step / num_steps_annealing) * math.pi)
            )
        )
        if self.global_step < num_steps_annealing:
            curr_temp = curr_temp_cos
        else:
            curr_temp = final_temp
        # curr_temp = torch.cos((self.global_step / num_steps_annealing) * math.pi / 2)
        # rate = (math.log(final_temp + 1e-10) - math.log(init_temp + 1e-10)) / float(num_steps_annealing)
        # curr_temp = max(init_temp * math.exp(rate * self.global_step), final_temp)
        return curr_temp

    def linear_annealing(
        self, init_temp=None, final_temp=None, num_steps_annealing=None
    ):
        if init_temp is None:
            init_temp = self.cfg.init_temp
        if final_temp is None:
            final_temp = self.cfg.model.final_temp
        if num_steps_annealing is None:
            num_steps_annealing = self.cfg.model.num_steps_annealing

        if self.global_step < num_steps_annealing:
            curr_temp = (1 - self.global_step / num_steps_annealing) * init_temp + (
                self.global_step / num_steps_annealing
            ) * final_temp

        else:
            curr_temp = final_temp
        return curr_temp

    def assign_label_names(self, label_names):
        self.label_names = label_names
