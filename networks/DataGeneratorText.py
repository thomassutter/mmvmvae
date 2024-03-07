import torch.nn as nn

from networks.ResidualBlocks import ResidualBlock1dTransposeConv


def res_block_decoder(
    in_channels,
    out_channels,
    kernelsize,
    stride,
    padding,
    o_padding,
    dilation,
    a_val=1.0,
    b_val=1.0,
):
    upsample = None

    if (
        (kernelsize != 1 or stride != 1)
        or (in_channels != out_channels)
        or dilation != 1
    ):
        upsample = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=kernelsize,
                stride=stride,
                padding=padding,
                dilation=dilation,
                output_padding=o_padding,
            ),
            nn.BatchNorm1d(out_channels),
        )
    layers = []
    layers.append(
        ResidualBlock1dTransposeConv(
            in_channels,
            out_channels,
            kernelsize,
            stride,
            padding,
            dilation,
            o_padding,
            upsample=upsample,
            a=a_val,
            b=b_val,
        )
    )
    return nn.Sequential(*layers)


class DataGeneratorText(nn.Module):
    def __init__(self, cfg, a, b):
        super(DataGeneratorText, self).__init__()
        self.cfg = cfg
        self.resblock_1 = res_block_decoder(
            5 * cfg.dataset.filter_dim_text,
            5 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=1,
            padding=0,
            dilation=1,
            o_padding=0,
            a_val=a,
            b_val=b,
        )
        self.resblock_2 = res_block_decoder(
            5 * cfg.dataset.filter_dim_text,
            5 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            o_padding=0,
            a_val=a,
            b_val=b,
        )
        self.resblock_3 = res_block_decoder(
            5 * cfg.dataset.filter_dim_text,
            4 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            o_padding=0,
            a_val=a,
            b_val=b,
        )
        self.resblock_4 = res_block_decoder(
            4 * cfg.dataset.filter_dim_text,
            3 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            o_padding=0,
            a_val=a,
            b_val=b,
        )
        self.resblock_5 = res_block_decoder(
            3 * cfg.dataset.filter_dim_text,
            2 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            o_padding=0,
            a_val=a,
            b_val=b,
        )
        self.resblock_6 = res_block_decoder(
            2 * cfg.dataset.filter_dim_text,
            cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            o_padding=0,
            a_val=a,
            b_val=b,
        )
        self.conv2 = nn.ConvTranspose1d(
            cfg.dataset.filter_dim_text,
            cfg.dataset.num_features,
            kernel_size=4,
            stride=2,
            padding=1,
            dilation=1,
            output_padding=0,
        )

    def forward(self, feats):
        d = self.resblock_1(feats)
        d = self.resblock_2(d)
        d = self.resblock_3(d)
        d = self.resblock_4(d)
        d = self.resblock_5(d)
        d = self.resblock_6(d)
        d = self.conv2(d)
        return d
