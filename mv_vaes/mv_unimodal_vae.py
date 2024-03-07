import torch
from mv_vaes.mv_vae import MVVAE


class MVunimodalVAE(MVVAE):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_hyperparameters()

    def log_additional_values(self, out):
        pass

    def log_additional_values_val(self):
        pass

    def forward(self, batch):
        data = batch[0]

        # encode views
        mus = []
        lvs = []
        dists_enc_out = {}
        # decode views
        dists_out = {}
        mods_rec = {}
        for m, key in enumerate(self.modality_names):
            # encode
            mod_m = data[key]
            mu_m, lv_m = self.encoders[m](mod_m)
            dists_enc_out[key] = [mu_m, lv_m]
            mus.append(mu_m.unsqueeze(1))
            lvs.append(lv_m.unsqueeze(1))
            z_m_out = self.reparametrize(mu_m, lv_m)

            # decode
            mod_hat_m = self.decoders[m](z_m_out)
            mods_rec[key] = mod_hat_m
            dist_out_m = [mu_m, lv_m]
            dists_out[key] = dist_out_m
        mus = torch.cat(mus, dim=1)
        lvs = torch.cat(lvs, dim=1)
        return (mods_rec, dists_out, dists_enc_out)

    def compute_loss(self, str_set, batch, forward_out):
        data, _ = batch
        data_rec = forward_out[0]
        dists_out = forward_out[1]

        # kl divergence of latent distribution
        klds = []
        for m, key in enumerate(self.modality_names):
            dist_m = dists_out[key]
            kld_m = self.kl_div_z(dist_m)
            klds.append(self.betas[key] * kld_m.unsqueeze(1))
            self.log(
                f"{str_set}/loss/kld_{key}",
                kld_m.mean(dim=0),
            )
        klds_sum = torch.cat(klds, dim=1).sum(dim=1)

        ## compute reconstruction loss/ conditional log-likelihood out data
        ## given latents
        loss_rec, loss_rec_mods, loss_rec_mods_weighted = self.compute_rec_loss(
            data, data_rec
        )
        for m, key in enumerate(self.modality_names):
            self.log(
                f"{str_set}/loss/weighted_rec_loss_{key}",
                loss_rec_mods_weighted[key],
            )
            self.log(
                f"{str_set}/loss/rec_loss_{key}",
                loss_rec_mods[key],
            )

        beta = self.cfg.model.beta
        loss_mv_vae = (loss_rec + beta * klds_sum).mean(dim=0)
        total_loss = loss_mv_vae
        # logging
        self.log(str_set + "/loss/klds_avg", klds_sum.mean(dim=0))
        self.log(str_set + "/loss/loss_rec", loss_rec.mean(dim=0))
        self.log(str_set + "/loss/mv_vae", loss_mv_vae)
        self.log(str_set + "/loss/loss", total_loss)
        return total_loss, loss_rec
