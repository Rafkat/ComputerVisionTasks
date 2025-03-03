import torch
import torch.nn as nn
from timm.models.layers import DropPath


# originated from https://arxiv.org/pdf/2201.03545


class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, drop_path=0., layer_scale_init_value=1e-6):
        super(ConvNextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.conv2 = nn.Linear(in_channels, 4 * in_channels)
        self.gelu = nn.GELU()
        self.conv3 = nn.Linear(4 * in_channels, in_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channels,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_tensor = x
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        x = input_tensor + x
        return x


class ConvNext(nn.Module):
    def __init__(self, channels, blocks_config, num_classes=1000,
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super(ConvNext, self).__init__()
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=4, stride=4)
        self.norm1 = nn.LayerNorm(channels[0])

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(channels))]

        self.downsample_layer_norm = nn.ModuleList(
            nn.LayerNorm(channels[i]) for i in range(len(channels) - 1)
        )

        self.downsample_conv = nn.ModuleList(
            nn.Conv2d(channels[i], channels[i + 1], kernel_size=2, stride=2) for i in range(len(channels) - 1)
        )

        self.first_block = nn.Sequential(
            *[ConvNextBlock(channels[0], drop_path=dp_rates[0 + i],
                            layer_scale_init_value=layer_scale_init_value)
              for i in range(blocks_config[0])]
        )

        self.second_block = nn.Sequential(
            *[ConvNextBlock(channels[1], drop_path=dp_rates[channels[0] + i],
                            layer_scale_init_value=layer_scale_init_value)
              for i in range(blocks_config[1])]
        )

        self.third_block = nn.Sequential(
            *[ConvNextBlock(channels[2], drop_path=dp_rates[sum(channels[:2]) + i],
                            layer_scale_init_value=layer_scale_init_value)
              for i in range(blocks_config[2])]
        )

        self.fourth_block = nn.Sequential(
            *[ConvNextBlock(channels[3], drop_path=dp_rates[sum(channels[:3]) + i],
                            layer_scale_init_value=layer_scale_init_value)
              for i in range(blocks_config[3])]
        )

        self.blocks = nn.ModuleList([self.first_block, self.second_block, self.third_block, self.fourth_block])

        self.norm2 = nn.LayerNorm(channels[-1])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        for i in range(len(self.downsample_layer_norm)):
            x = self.blocks[i](x)
            x = x.permute(0, 2, 3, 1)
            x = self.downsample_layer_norm[i](x)
            x = x.permute(0, 3, 1, 2)
            x = self.downsample_conv[i](x)
        x = self.blocks[-1](x)
        x = self.avg_pool(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
