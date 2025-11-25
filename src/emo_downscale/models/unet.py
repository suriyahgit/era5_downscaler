# src/emo_downscale/models/unet.py

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from emo_downscale.logging_utils import get_logger

logger = get_logger("models.unet")


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def get_norm(
    num_channels: int,
    norm: Literal["group", "batch", "none"],
    norm_groups: int = 8,
) -> nn.Module:
    norm = norm.lower()
    if norm == "group":
        # clamp groups so it always divides num_channels
        groups = min(norm_groups, num_channels)
        # fallback to 1 group if not divisible
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {norm}")


class DoubleConv(nn.Module):
    """
    Two (or more) conv-norm-activation layers in a row.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        convs_per_block: int = 2,
        norm: str = "group",
        norm_groups: int = 8,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        act = get_activation(activation)

        c_in = in_channels
        for i in range(convs_per_block):
            c_out = out_channels
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False))
            layers.append(get_norm(c_out, norm, norm_groups))
            layers.append(act)
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            c_in = c_out

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """
    A single encoder stage: DoubleConv + optional extra ops.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        convs_per_block: int = 2,
        norm: str = "group",
        norm_groups: int = 8,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            convs_per_block=convs_per_block,
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up(nn.Module):
    """
    A single decoder stage:
      upsample -> concat skip -> DoubleConv
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        upsample_mode: str = "bilinear",
        convs_per_block: int = 2,
        norm: str = "group",
        norm_groups: int = 8,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        upsample_mode = upsample_mode.lower()
        if upsample_mode == "bilinear":
            self.upsample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            )
        elif upsample_mode == "nearest":
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        elif upsample_mode == "transposed":
            # ConvTranspose2d will both upsample and reduce channels
            self.upsample = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
            in_channels = out_channels  # after upsampling
        else:
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")

        self.use_transposed = upsample_mode == "transposed"

        # After upsample we concat with skip along channels
        self.conv = DoubleConv(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            convs_per_block=convs_per_block,
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # if U is pure upsample and shapes don't match perfectly, fix spatial size
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Flexible 2D UNet with configurable depth, normalization, activation, and
    upsampling strategy. Designed to match the config in configs/emo1_unet.yaml.

    Args:
        in_channels:   Number of input channels (predictor variables).
        out_channels:  Number of output channels (target variables).
        base_channels: Number of filters in the first encoder block.
        num_down_blocks: Number of encoder/decoder stages.
        convs_per_block: Number of conv layers in each block.
        norm:          "group", "batch", or "none".
        norm_groups:   Number of groups for GroupNorm.
        activation:    "relu", "leaky_relu", "gelu", ...
        dropout:       Dropout probability for Conv blocks.
        upsample_mode: "bilinear", "nearest", or "transposed".
        residual_output: If True and shapes match, add input to output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        num_down_blocks: int = 4,
        convs_per_block: int = 2,
        norm: str = "group",
        norm_groups: int = 8,
        activation: str = "relu",
        dropout: float = 0.0,
        upsample_mode: str = "bilinear",
        residual_output: bool = False,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logger.warning(f"UNet received unused kwargs: {list(kwargs.keys())}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_down_blocks = num_down_blocks
        self.residual_output = residual_output

        # ---------------------------------------------------------------------
        # Encoder: Down blocks + pooling
        # ---------------------------------------------------------------------
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        # channel sizes for encoder stages, excluding bottleneck
        # e.g. [in, 64, 128, 256, 512] for num_down_blocks=4, base=64
        enc_channels = [in_channels]
        for i in range(num_down_blocks):
            enc_channels.append(base_channels * (2**i))

        for i in range(num_down_blocks):
            self.down_blocks.append(
                Down(
                    in_channels=enc_channels[i],
                    out_channels=enc_channels[i + 1],
                    convs_per_block=convs_per_block,
                    norm=norm,
                    norm_groups=norm_groups,
                    activation=activation,
                    dropout=dropout,
                )
            )
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # ---------------------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------------------
        bottleneck_in = enc_channels[-1]
        bottleneck_out = bottleneck_in * 2  # standard UNet pattern
        self.bottleneck = DoubleConv(
            in_channels=bottleneck_in,
            out_channels=bottleneck_out,
            convs_per_block=convs_per_block,
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
            dropout=dropout,
        )

        # ---------------------------------------------------------------------
        # Decoder: Up blocks
        # ---------------------------------------------------------------------
        self.up_blocks = nn.ModuleList()

        # build decoder from deepest to shallowest
        curr_channels = bottleneck_out
        for i in reversed(range(num_down_blocks)):
            skip_channels = enc_channels[i + 1]  # matching encoder stage
            out_ch = skip_channels

            self.up_blocks.append(
                Up(
                    in_channels=curr_channels,
                    skip_channels=skip_channels,
                    out_channels=out_ch,
                    upsample_mode=upsample_mode,
                    convs_per_block=convs_per_block,
                    norm=norm,
                    norm_groups=norm_groups,
                    activation=activation,
                    dropout=dropout,
                )
            )
            curr_channels = out_ch

        # ---------------------------------------------------------------------
        # Final 1x1 conv to map to desired output channels
        # ---------------------------------------------------------------------
        self.conv_last = nn.Conv2d(curr_channels, out_channels, kernel_size=1)

        logger.info(
            f"Initialized UNet: in={in_channels}, out={out_channels}, "
            f"base={base_channels}, depth={num_down_blocks}, "
            f"upsample_mode={upsample_mode}, residual_output={residual_output}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        returns: (B, C_out, H, W)
        """
        input_for_residual = x

        # ----------------- Encoder -----------------
        enc_features = []
        for down, pool in zip(self.down_blocks, self.pools):
            x = down(x)
            enc_features.append(x)
            x = pool(x)

        # ----------------- Bottleneck --------------
        x = self.bottleneck(x)

        # ----------------- Decoder -----------------
        # use stored encoder features in reverse
        for up, skip in zip(self.up_blocks, reversed(enc_features)):
            x = up(x, skip)

        out = self.conv_last(x)

        # optional residual output
        if self.residual_output:
            if (
                input_for_residual.shape[1] == out.shape[1]
                and input_for_residual.shape[-2:] == out.shape[-2:]
            ):
                out = out + input_for_residual
            else:
                # we only warn once to avoid log spam
                if not hasattr(self, "_residual_warned"):
                    logger.warning(
                        "UNet residual_output=True, but input and output shapes "
                        "do not match. Skipping residual addition."
                    )
                    self._residual_warned = True

        return out
