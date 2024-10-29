from torch import nn
from config.MyMVWSLConfig import MyMVWSLConfig

from networks.NetworksImgCelebA import EncoderImg, DecoderImg
from networks.NetworksTextCelebA import EncoderText, DecoderText
from networks.ConvNetworksPolyMNIST import Encoder, Decoder
from networks.ConvNetworksPolyMNIST import ResnetEncoder, ResnetDecoder
from networks.NetworksCUBimg import Encoder as EncoderCUBimg
from networks.NetworksCUBimg import Decoder as DecoderCUBimg
from networks.NetworksCUBsent import Encoder as EncoderCUBtext
from networks.NetworksCUBsent import Decoder as DecoderCUBtext
from networks.NetworksRatsspike import Encoder as RatsEncoder
from networks.NetworksRatsspike import Decoder as RatsDecoder


def get_networks(cfg: MyMVWSLConfig) -> list[nn.ModuleList]:
    if cfg.dataset.name.startswith("PM"):
        if not cfg.model.use_resnets:
            encoders = nn.ModuleList(
                [
                    Encoder(cfg.model.latent_dim).to(cfg.model.device)
                    for _ in range(cfg.dataset.num_views)
                ]
            )
            decoders = nn.ModuleList(
                [
                    Decoder(cfg.model.latent_dim).to(cfg.model.device)
                    for _ in range(cfg.dataset.num_views)
                ]
            )
        else:
            encoders = nn.ModuleList(
                [
                    ResnetEncoder(cfg).to(cfg.model.device)
                    for _ in range(cfg.dataset.num_views)
                ]
            )
            decoders = nn.ModuleList(
                [
                    ResnetDecoder(cfg).to(cfg.model.device)
                    for _ in range(cfg.dataset.num_views)
                ]
            )
    elif cfg.dataset.name.startswith("celeba"):
        encoders = nn.ModuleList(
            [
                EncoderImg(cfg).to(cfg.model.device),
                EncoderText(cfg).to(cfg.model.device),
            ]
        )
        decoders = nn.ModuleList(
            [
                DecoderImg(cfg).to(cfg.model.device),
                DecoderText(cfg).to(cfg.model.device),
            ]
        )
    elif cfg.dataset.name.startswith("CUB"):
        print("add encoders and decoder for cub...")
        encoders = nn.ModuleList(
            [
                EncoderCUBtext(cfg.model.latent_dim).to(cfg.model.device),
                EncoderCUBimg(cfg.model.latent_dim).to(cfg.model.device),
            ]
        )
        decoders = nn.ModuleList(
            [
                DecoderCUBtext(cfg.model.latent_dim).to(cfg.model.device),
                DecoderCUBimg(cfg.model.latent_dim).to(cfg.model.device),
            ]
        )
    elif cfg.dataset.name.startswith(""):
        original_dims = [92, 79, 104, 49, 46]
        encoders = nn.ModuleList(
            [
                RatsEncoder(cfg.model.latent_dim, original_dims[m]).to(cfg.model.device)
                for m in range(cfg.dataset.num_views)
            ]
        )
        decoders = nn.ModuleList(
            [
                RatsDecoder(cfg.model.latent_dim, original_dims[m]).to(cfg.model.device)
                for m in range(cfg.dataset.num_views)
            ]
        )
    else:
        raise NotImplementedError(
            "Unknown dataset/networks to create encoders and decoders for specified config"
        )
    return [encoders, decoders]
