import torch
import torch.nn as nn


# originated from https://arxiv.org/pdf/1704.04861v1

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, groups=in_channels,
                                   padding=kernel_size // 2)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self, deep_blocks_num, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.sep_conv1 = DepthwiseSeparableConv(32, 64, 3, 1)

        self.sep_conv2 = DepthwiseSeparableConv(64, 128, 3, 2)

        self.sep_conv3 = DepthwiseSeparableConv(128, 128, 3, 1)

        self.sep_conv4 = DepthwiseSeparableConv(128, 256, 3, 2)

        self.sep_conv5 = DepthwiseSeparableConv(256, 256, 3, 1)

        self.sep_conv6 = DepthwiseSeparableConv(256, 512, 3, 2)

        self.deep_blocks = nn.Sequential(*[DepthwiseSeparableConv(512, 512, 3, 1)
                                           for _ in range(deep_blocks_num)])

        self.sep_conv7 = DepthwiseSeparableConv(512, 1024, 3, 2)

        self.sep_conv8 = DepthwiseSeparableConv(1024, 1024, 3, 1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)
        x = self.sep_conv4(x)
        x = self.sep_conv5(x)
        x = self.sep_conv6(x)
        x = self.deep_blocks(x)
        x = self.sep_conv7(x)
        x = self.sep_conv8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = MobileNetV1(deep_blocks_num=5)
    model(torch.randn(1, 3, 224, 224))