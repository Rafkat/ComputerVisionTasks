import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50


# originated from https://arxiv.org/pdf/1612.01105


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PyramidPooling(nn.Module):
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pooling_size = [1, 2, 3, 6]
        self.channels = in_channels // 4

        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[0]),
            ConvBlock(in_channels, self.channels)
        )

        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[1]),
            ConvBlock(in_channels, self.channels)
        )

        self.pool3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[2]),
            ConvBlock(in_channels, self.channels)
        )

        self.pool4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[3]),
            ConvBlock(in_channels, self.channels)
        )

    @staticmethod
    def upsample(input_tensor, size=None, scale_factor=None, align_corners=None):
        out = F.interpolate(input_tensor,
                            size=size,
                            scale_factor=scale_factor,
                            align_corners=align_corners,
                            mode='bilinear')
        return out

    def forward(self, x):
        out1 = self.pool1(x)
        out1 = self.upsample(out1, size=x.size()[-2:])

        out2 = self.pool2(x)
        out2 = self.upsample(out2, size=x.size()[-2:])

        out3 = self.pool3(x)
        out3 = self.upsample(out3, size=x.size()[-2:])

        out4 = self.pool4(x)
        out4 = self.upsample(out4, size=x.size()[-2:])

        return torch.cat([out1, out2, out3, out4], dim=1)


class PSPNet(nn.Module):
    def __init__(self, nb_classes=21):
        super(PSPNet, self).__init__()
        self.out_channels = 2048

        self.backbone = resnet50(weights='DEFAULT')
        self.feature_extractor = nn.Sequential(
            *list(self.backbone.children())[:-2]
        )

        self.depth = self.out_channels // 4
        self.pyramid_pooling = PyramidPooling(self.out_channels)

        self.decoder = nn.Sequential(
            ConvBlock(self.out_channels * 2, self.depth),
            nn.Dropout(0.1),
            nn.Conv2d(self.depth, nb_classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.size()[-2:]
        feature_map = self.feature_extractor(x)

        x = self.pyramid_pooling(feature_map)
        x = torch.concat([feature_map, x], dim=1)
        x = self.decoder(x)
        x = PyramidPooling.upsample(x, size=(h, w), align_corners=True)
        return x


if __name__ == '__main__':
    model = PSPNet()
    model.eval()
    model(torch.randn(1, 3, 224, 224))
