import torch.nn as nn

from networks.ResidualBlocks import ResidualBlock1dConv


def make_res_block_encoder_feature_extractor(
    in_channels,
    out_channels,
    kernelsize,
    stride,
    padding,
    dilation,
    a_val=1.0,
    b_val=1.0,
):
    downsample = None
    if (stride != 1) or (in_channels != out_channels) or dilation != 1:
        downsample = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernelsize,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
        )
    layers = []
    layers.append(
        ResidualBlock1dConv(
            in_channels,
            out_channels,
            kernelsize,
            stride,
            padding,
            dilation,
            downsample,
            a=a_val,
            b=b_val,
        )
    )
    return nn.Sequential(*layers)


class FeatureExtractorText(nn.Module):
    def __init__(self, cfg, a, b):
        super(FeatureExtractorText, self).__init__()
        self.a = a
        self.b = b
        self.conv1 = nn.Conv1d(
            cfg.dataset.num_features,
            cfg.dataset.filter_dim_text,
            kernel_size=4,
            stride=2,
            padding=1,
            dilation=1,
        )
        self.resblock_1 = make_res_block_encoder_feature_extractor(
            cfg.dataset.filter_dim_text,
            2 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=a,
            b_val=b,
        )
        self.resblock_2 = make_res_block_encoder_feature_extractor(
            2 * cfg.dataset.filter_dim_text,
            3 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=a,
            b_val=b,
        )
        self.resblock_3 = make_res_block_encoder_feature_extractor(
            3 * cfg.dataset.filter_dim_text,
            4 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=a,
            b_val=b,
        )
        self.resblock_4 = make_res_block_encoder_feature_extractor(
            4 * cfg.dataset.filter_dim_text,
            5 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=a,
            b_val=b,
        )
        self.resblock_5 = make_res_block_encoder_feature_extractor(
            5 * cfg.dataset.filter_dim_text,
            5 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=1,
            dilation=1,
            a_val=a,
            b_val=b,
        )
        self.resblock_6 = make_res_block_encoder_feature_extractor(
            5 * cfg.dataset.filter_dim_text,
            5 * cfg.dataset.filter_dim_text,
            kernelsize=4,
            stride=2,
            padding=0,
            dilation=1,
            a_val=a,
            b_val=b,
        )

    def forward(self, x):
        x = x.transpose(-2, -1)
        out = self.conv1(x)
        out = self.resblock_1(out)
        out = self.resblock_2(out)
        out = self.resblock_3(out)
        out = self.resblock_4(out)
        out = self.resblock_5(out)
        out = self.resblock_6(out)
        return out
