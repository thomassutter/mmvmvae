import torch
from torch import nn
from mv_vaes.mv_vae import MVVAE


# implements the mmvae+ multimodal vae
class MVSplitVAE(MVVAE):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.model.aggregation == "moe":
            self.aggregate_latents = self.aggregate_latents_moe
        elif cfg.model.aggregation == "poe":
            self.aggregate_latents = self.aggregate_latents_poe
        elif cfg.model.aggregation == "mopoe":
            self.aggregate_latents = self.aggregate_latents_mopoe
        else:
            self.aggregate_latents = self.aggregate_latents_avg
        self.save_hyperparameters()

        assert cfg.model.split_type in ["simple", "plus"]
        if cfg.model.split_type == "plus":
            self.lvs_ms = nn.ParameterDict()
            self.mus_ms = nn.ParameterDict()
            for key in self.modality_names:
                self.lvs_ms[key] = nn.Parameter(
                    torch.zeros(
                        size=(1, cfg.model.mod_specific_dim),
                        device=cfg.model.device,
                    )
                )
                self.mus_ms[key] = nn.Parameter(
                    torch.zeros(
                        size=(1, cfg.model.mod_specific_dim),
                        device=cfg.model.device,
                    )
                )

    def log_additional_values(self, out):
        pass

    def log_additional_values_val(self):
        pass

    def forward(self, batch):
        if self.cfg.model.split_type == "simple":
            out_f = self.forward_simple(batch)
        elif self.cfg.model.split_type == "plus":
            out_f = self.forward_plus(batch)
        else:
            raise NotImplementedError
        return out_f

    def forward_simple(self, batch):
        data = batch[0]

        mus = []
        lvs = []
        dists_enc_out = {}
        dists_enc_ms_out = {}
        # encode views
        # for m in range(0, self.cfg.dataset.num_views):
        for m, key in enumerate(self.modality_names):
            mod_m = data[key]
            mu_m, lv_m = self.encoders[m](mod_m)
            mu_m_s, mu_m_ms = torch.split(
                mu_m,
                [
                    self.cfg.model.latent_dim - self.cfg.model.mod_specific_dim,
                    self.cfg.model.mod_specific_dim,
                ],
                dim=-1,
            )
            lv_m_s, lv_m_ms = torch.split(
                lv_m,
                [
                    self.cfg.model.latent_dim - self.cfg.model.mod_specific_dim,
                    self.cfg.model.mod_specific_dim,
                ],
                dim=-1,
            )
            mus.append(mu_m_s.unsqueeze(1))
            lvs.append(lv_m_s.unsqueeze(1))
            dists_enc_out[key] = [mu_m_s, lv_m_s]
            dists_enc_ms_out[key] = [mu_m_ms, lv_m_ms]
        mus = torch.cat(mus, dim=1)
        lvs = torch.cat(lvs, dim=1)

        mu_out, lv_out = self.aggregate_latents(mus, lvs)
        z_out = self.reparametrize(mu_out, lv_out)

        # decode views
        dists_out = {}
        mods_rec = {}
        # for m in range(0, self.cfg.dataset.num_views):
        for m, key in enumerate(self.modality_names):
            mu_m_ms, lv_m_ms = dists_enc_ms_out[key]
            z_ms_out = self.reparametrize(mu_m_ms, lv_m_ms)
            mod_hat_m = self.decoders[m](torch.cat([z_out, z_ms_out], dim=1))
            mods_rec[key] = mod_hat_m

            dists_out[key] = [mu_out, lv_out]
        return (mods_rec, dists_out, dists_enc_out, dists_enc_ms_out)

    def forward_plus(self, batch):
        data = batch[0]

        mus = []
        lvs = []
        dists_enc_out = {}
        dists_enc_ms_out = {}
        mods_rec = {key: {} for key in self.modality_names}
        # encode views
        # for m in range(0, self.cfg.dataset.num_views):
        for m, key in enumerate(self.modality_names):
            mod_m = data[key]
            mu_m, lv_m = self.encoders[m](mod_m)
            z_out_m = self.reparametrize(mu_m, lv_m)
            z_out_m_s, z_out_m_ms = torch.split(
                z_out_m,
                [
                    self.cfg.model.latent_dim - self.cfg.model.mod_specific_dim,
                    self.cfg.model.mod_specific_dim,
                ],
                dim=-1,
            )

            mu_m_s, mu_m_ms = torch.split(
                mu_m,
                [
                    self.cfg.model.latent_dim - self.cfg.model.mod_specific_dim,
                    self.cfg.model.mod_specific_dim,
                ],
                dim=-1,
            )
            lv_m_s, lv_m_ms = torch.split(
                lv_m,
                [
                    self.cfg.model.latent_dim - self.cfg.model.mod_specific_dim,
                    self.cfg.model.mod_specific_dim,
                ],
                dim=-1,
            )
            mus.append(mu_m_s.unsqueeze(1))
            lvs.append(lv_m_s.unsqueeze(1))
            dists_enc_out[key] = [mu_m_s, lv_m_s]
            dists_enc_ms_out[key] = [mu_m_ms, lv_m_ms]
            for m_tilde, key_tilde in enumerate(self.modality_names):
                if m == m_tilde:
                    mod_hat_m_mtilde = self.decoders[m_tilde](z_out_m)
                else:
                    mu_mtilde = self.mus_ms[key_tilde]
                    lv_mtilde = self.lvs_ms[key_tilde].repeat(z_out_m_s.shape[0], 1)
                    z_out_mtilde_ms = self.reparametrize(mu_mtilde, lv_mtilde)
                    mod_hat_m_mtilde = self.decoders[m_tilde](
                        torch.cat([z_out_m_s, z_out_mtilde_ms], dim=1)
                    )
                mods_rec[key][key_tilde] = mod_hat_m_mtilde

        mus = torch.cat(mus, dim=1)
        lvs = torch.cat(lvs, dim=1)
        mu_out, lv_out = self.aggregate_latents(mus, lvs)
        dists_out = {}
        for key in self.modality_names:
            dists_out[key] = [mu_out, lv_out]
        return (mods_rec, dists_out, dists_enc_out, dists_enc_ms_out)

    def cond_generate_samples(self, m, z):
        z_ms = torch.randn(
            (z.shape[0], self.cfg.model.mod_specific_dim), device=self.cfg.model.device
        )
        mod_c_gen_m_tilde = self.decoders[m](torch.cat([z, z_ms], dim=1))
        return mod_c_gen_m_tilde

    def get_reconstructions(self, mods_out, key, n_samples):
        if self.cfg.model.split_type == "simple":
            mod_rec = self.get_reconstructions_simple(mods_out, key, n_samples)
        if self.cfg.model.split_type == "plus":
            mod_rec = self.get_reconstructions_plus(mods_out, key, n_samples)
        return mod_rec

    def get_reconstructions_plus(self, mods_out, key, n_samples):
        mod_rec = mods_out[key][key][0][:n_samples]
        return mod_rec

    def get_reconstructions_simple(self, mods_out, key, n_samples):
        mod_rec = mods_out[key][0][:n_samples]
        return mod_rec

    def compute_rec_loss_simple(self, data, mods_rec, str_set=None):
        loss_rec, loss_rec_mods, loss_rec_mods_weighted = self.compute_rec_loss(
            data, mods_rec
        )
        return loss_rec

    def compute_rec_loss_plus(self, data, mods_rec, str_set=None):
        ## compute reconstruction loss/ conditional log-likelihood out data
        ## given latents
        losses_rec = []
        for key in self.modality_names:
            (
                loss_rec_m,
                loss_rec_mods_m,
                loss_rec_mods_weighted_m,
            ) = self.compute_rec_loss(data, mods_rec[key])
            losses_rec.append(loss_rec_m.unsqueeze(1))
            self.log(
                f"{str_set}/loss/weighted_rec_loss_{key}",
                sum(loss_rec_mods_weighted_m.values()),
            )
            self.log(
                f"{str_set}/loss/rec_loss_{key}",
                sum(loss_rec_mods_m.values()),
            )
        losses_rec = torch.cat(losses_rec, dim=1)
        loss_rec = torch.mean(losses_rec, dim=1)
        return loss_rec

    def compute_loss(self, str_set, batch, forward_out):
        data, _ = batch
        data_rec = forward_out[0]
        dists_out = forward_out[1]
        dists_enc_ms_out = forward_out[3]

        # kl divergence of latent distribution
        klds_shared = []
        klds_ms = []
        for key in self.modality_names:
            dist_m_s = dists_out[key]
            kld_m = self.kl_div_z(dist_m_s)
            klds_shared.append(kld_m.unsqueeze(1))
            dist_m_ms = dists_enc_ms_out[key]
            kld_m_ms = self.kl_div_z(dist_m_ms)
            klds_ms.append(kld_m_ms.unsqueeze(1))
        klds_shared_sum = torch.cat(klds_shared, dim=1).sum(dim=1)
        klds_ms_sum = torch.cat(klds_ms, dim=1).sum(dim=1)
        klds_sum = klds_shared_sum + klds_ms_sum

        if self.cfg.model.split_type == "simple":
            loss_rec = self.compute_rec_loss_simple(data, data_rec, str_set)
        if self.cfg.model.split_type == "plus":
            loss_rec = self.compute_rec_loss_plus(data, data_rec, str_set)

        beta = self.cfg.model.beta
        loss_mv_vae = (loss_rec + beta * klds_sum).mean(dim=0)
        total_loss = loss_mv_vae
        # logging
        self.log(str_set + "/loss/klds_avg", klds_sum.mean(dim=0))
        self.log(str_set + "/loss/klds_avg_shared", klds_shared_sum.mean(dim=0))
        self.log(str_set + "/loss/klds_avg_ms", klds_ms_sum.mean(dim=0))
        self.log(str_set + "/loss/loss_rec", loss_rec.mean(dim=0))
        self.log(str_set + "/loss/mv_vae", loss_mv_vae)
        self.log(str_set + "/loss/loss", total_loss)
        return total_loss, loss_rec
