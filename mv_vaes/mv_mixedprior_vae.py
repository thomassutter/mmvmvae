import torch
from mv_vaes.mv_vae import MVVAE


class MVMixedPriorVAE(MVVAE):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_hyperparameters()

    def log_additional_values(self, out):
        pass

    def log_additional_values_val(self):
        pass

    def forward(self, batch):
        data = batch[0]

        dists_enc_out = {}
        dists_out = {}
        mods_rec = {}
        for m, key in enumerate(data.keys()):
            # encode views: img_m -> z_m
            mod_m = data[key]
            mu_m, lv_m = self.encoders[m](mod_m)
            dists_enc_out[key] = [mu_m, lv_m]
            z_m = self.reparametrize(mu_m, lv_m)

            # decode views: z_m -> img_hat_m
            mod_hat_m = self.decoders[m](z_m)
            mods_rec[key] = mod_hat_m

            dist_out_m = [mu_m, lv_m]
            dists_out[key] = dist_out_m
        return (mods_rec, dists_out, dists_enc_out)

    def compute_loss(self, str_set, batch, forward_out):
        imgs, _ = batch
        imgs_rec = forward_out[0]
        dists_out = forward_out[1]
        priors = dists_out

        if self.cfg.model.alpha_annealing:
            init_temp = self.cfg.model.init_alpha_value
            final_temp = self.cfg.model.final_alpha_value
            annealing_steps = self.cfg.model.alpha_annealing_steps
            alpha_weight = self.compute_current_temperature(
                init_temp, final_temp, annealing_steps
            )
        else:
            alpha_weight = self.cfg.model.final_alpha_value
        self.log("alpha annealing", alpha_weight)
        klds = []
        for key in self.modality_names:
            dist_m = dists_out[key]
            for key_tilde in self.modality_names:
                dist_m_tilde = priors[key_tilde]
                kld_m_m_tilde = self.kl_div_z_two_dists(dist_m, dist_m_tilde)
                # KL(q_m | q_m_tilde) * (1-alpha)
                klds.append(kld_m_m_tilde.unsqueeze(1) * (1.0 - alpha_weight))
            # add N(0,1) as a component
            kld_m = self.kl_div_z(dist_m)
            # KL(q_m | N(0,1)) * alpha * M
            klds.append(kld_m.unsqueeze(1) * alpha_weight * self.cfg.dataset.num_views)
        # SUM_{m}:( alpha * KL(q_m|N(0,1)) + (1-alpha)/M * SUM_{m_tilde}:KL(q_m|q_m_tilde) )
        # when alpha = 0: mixedprior
        # when alpha = 1: unimodal
        # when alpha = 1/(M+1): mixedpriorstdnorm
        klds_sum = torch.cat(klds, dim=1).sum(dim=1) / self.cfg.dataset.num_views

        ## compute reconstruction loss/ conditional log-likelihood out data
        ## given latents
        loss_rec, loss_rec_mods, loss_rec_mods_weighted = self.compute_rec_loss(
            imgs, imgs_rec
        )
        for key in self.modality_names:
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
