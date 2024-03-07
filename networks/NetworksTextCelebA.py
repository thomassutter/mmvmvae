import torch.nn as nn

from networks.FeatureExtractorText import FeatureExtractorText
from networks.FeatureCompressor import LinearFeatureCompressor
from networks.DataGeneratorText import DataGeneratorText


class EncoderText(nn.Module):
    def __init__(self, cfg):
        super(EncoderText, self).__init__()
        self.feature_extractor = FeatureExtractorText(
            cfg,
            a=cfg.dataset.skip_connections_text_weight_a,
            b=cfg.dataset.skip_connections_text_weight_b,
        )
        self.feature_compressor = LinearFeatureCompressor(
            5 * cfg.dataset.filter_dim_text, cfg.model.latent_dim
        )

    def forward(self, x_text):
        h_text = self.feature_extractor(x_text)
        mu, logvar = self.feature_compressor(h_text)
        # return mu, logvar, h_text;
        return mu, logvar


class DecoderText(nn.Module):
    def __init__(self, cfg):
        super(DecoderText, self).__init__()
        self.feature_generator = nn.Linear(
            cfg.model.latent_dim, 5 * cfg.dataset.filter_dim_text, bias=True
        )
        self.text_generator = DataGeneratorText(
            cfg,
            a=cfg.dataset.skip_connections_text_weight_a,
            b=cfg.dataset.skip_connections_text_weight_b,
        )

    def forward(self, z):
        text_feat_hat = self.feature_generator(z)
        text_feat_hat = text_feat_hat.unsqueeze(-1)
        text_hat = self.text_generator(text_feat_hat)
        text_hat = text_hat.transpose(-2, -1)
        return [text_hat]
