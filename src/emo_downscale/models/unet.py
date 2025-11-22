import torch
from torch import nn
from emo_downscale.logging_utils import get_logger
logger = get_logger("models.unet")



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=6,
        out_channels=1,
        base_channels=64,
        num_down_blocks=4,
        convs_per_block=2,
        **kwargs,
    ):
        super().__init__()

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        channels = [in_channels]
        for i in range(num_down_blocks):
            channels.append(base_channels * (2 ** i))

        # Encoder
        for i in range(num_down_blocks):
            self.down_blocks.append(DoubleConv(channels[i], channels[i + 1]))
            self.pools.append(nn.MaxPool2d(2))

        # Bottleneck
        self.bottleneck = DoubleConv(channels[-1], channels[-1] * 2)

        # Decoder
        for i in range(num_down_blocks):
            self.up_blocks.append(
                nn.ConvTranspose2d(channels[-1] * 2 // (2 ** i), channels[-1] // (2 ** i), 2, stride=2)
            )

        self.conv_last = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        enc_features = []

        for down, pool in zip(self.down_blocks, self.pools):
            x = down(x)
            enc_features.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, enc in zip(self.up_blocks[::-1], enc_features[::-1]):
            x = up(x)
            x = torch.cat([enc, x], dim=1)

        return self.conv_last(x)
