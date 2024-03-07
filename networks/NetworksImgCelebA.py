import torch
import torch.nn as nn

from networks.FeatureExtractorImg import FeatureExtractorImg
from networks.FeatureCompressor import LinearFeatureCompressor
from networks.DataGeneratorImg import DataGeneratorImg


class EncoderImg(nn.Module):
    def __init__(self, cfg):
        super(EncoderImg, self).__init__()
        self.feature_extractor = FeatureExtractorImg(
            cfg,
            a=cfg.dataset.skip_connections_img_weight_a,
            b=cfg.dataset.skip_connections_img_weight_b,
        )
        self.feature_compressor = LinearFeatureCompressor(
            cfg.dataset.num_layers_img * cfg.dataset.filter_dim_img,
            cfg.model.latent_dim,
        )

    def forward(self, x_img):
        h_img = self.feature_extractor(x_img)
        h_img = h_img.view(h_img.shape[0], h_img.shape[1], h_img.shape[2])
        mu, logvar = self.feature_compressor(h_img)
        return mu, logvar


class DecoderImg(nn.Module):
    def __init__(self, cfg):
        super(DecoderImg, self).__init__()
        self.feature_generator = nn.Linear(
            cfg.model.latent_dim,
            cfg.dataset.num_layers_img * cfg.dataset.filter_dim_img,
            bias=True,
        )
        self.img_generator = DataGeneratorImg(
            cfg,
            a=cfg.dataset.skip_connections_img_weight_a,
            b=cfg.dataset.skip_connections_img_weight_b,
        )

    def forward(self, z):
        img_feat_hat = self.feature_generator(z)
        img_feat_hat = img_feat_hat.view(
            img_feat_hat.size(0), img_feat_hat.size(1), 1, 1
        )
        img_hat = self.img_generator(img_feat_hat)
        return img_hat, torch.tensor(0.75).to(z.device)
