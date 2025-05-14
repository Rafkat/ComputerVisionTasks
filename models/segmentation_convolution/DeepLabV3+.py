import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor

# originated from https://arxiv.org/pdf/1706.05587


class AtrousConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation_rate):
        super(AtrousConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              dilation=dilation_rate)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AtrousSpatialPyramidPooling, self).__init__()
        self.conv_1x1 = AtrousConv(in_channels, out_channels, kernel_size=1, padding=0, dilation_rate=1)
        self.conv_6x6 = AtrousConv(in_channels, out_channels, kernel_size=3, padding=6, dilation_rate=6)
        self.conv_12x12 = AtrousConv(in_channels, out_channels, kernel_size=3, padding=12, dilation_rate=12)
        self.conv_18x18 = AtrousConv(in_channels, out_channels, kernel_size=3, padding=18, dilation_rate=18)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.final_conv = AtrousConv(out_channels * 5, out_channels, kernel_size=1, padding=0, dilation_rate=1)

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_6x6 = self.conv_6x6(x)
        x_12x12 = self.conv_12x12(x)
        x_18x18 = self.conv_18x18(x)
        img_pool_opt = self.image_pool(x)
        img_pool_opt = F.interpolate(img_pool_opt, size=x_18x18.size()[2:],
                                     mode='bilinear', align_corners=True)

        concat = torch.cat([x_1x1, x_6x6, x_12x12, x_18x18, img_pool_opt], dim=1)
        x_final_conv = self.final_conv(concat)
        return x_final_conv


class DeepLabV3Plus(nn.Module):
    def __init__(self, nb_classes):
        super(DeepLabV3Plus, self).__init__()
        high_level_out = 1024
        low_level_out = 256
        self.nb_classes = nb_classes

        self.backbone = resnet50(weights='DEFAULT')
        self.feat_ext = create_feature_extractor(
            self.backbone, {'layer1': 'low_level', 'layer3': 'high_level'}
        )

        self.aspp = AtrousSpatialPyramidPooling(high_level_out, out_channels=256)

        self.conv_1x1 = AtrousConv(low_level_out, 48, kernel_size=1, padding=0, dilation_rate=1)
        self.conv_3x3_1 = nn.Sequential(
            nn.Conv2d(304, 304, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(304),
            nn.ReLU()
        )
        self.conv_3x3_2 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.head = nn.Conv2d(256, nb_classes, 1)

    def forward(self, x):
        features = self.feat_ext(x)
        low_level = features.get('low_level')
        high_level = features.get('high_level')
        feature_map_conv = self.conv_1x1(low_level)

        aspp_res = self.aspp(high_level)
        aspp_res = F.interpolate(aspp_res, scale_factor=(4, 4), mode='bilinear', align_corners=True)

        x = torch.concat([feature_map_conv, aspp_res], dim=1)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    model = DeepLabV3Plus(10)
    model.eval()
    model(torch.randn(1, 3, 256, 256))