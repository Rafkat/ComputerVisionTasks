import torch
import torch.nn as nn


# originated from https://arxiv.org/pdf/1801.04381

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, expansion_factor * in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(expansion_factor * in_channels)
        self.relu1 = nn.ReLU6()

        self.conv2 = nn.Conv2d(expansion_factor * in_channels,
                               expansion_factor * in_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               groups=expansion_factor * in_channels)

        self.bn2 = nn.BatchNorm2d(expansion_factor * in_channels)
        self.relu2 = nn.ReLU6()

        self.conv3 = nn.Conv2d(expansion_factor * in_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.stride == 1 and out.size() == x.size():
            return out + x
        return out


class MobileNetV2(nn.Module):
    def __init__(self, nb_classes=10):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU6()

        self.basic_block1 = BasicBlock(32, 16, 1, 1)

        self.bottleneck_layer1 = nn.Sequential(
            BasicBlock(16, 24, 2, 6),
            BasicBlock(24, 24, 1, 6)
        )

        self.bottleneck_layer2 = nn.Sequential(
            BasicBlock(24, 32, 2, 6),
            BasicBlock(32, 32, 1, 6),
            BasicBlock(32, 32, 1, 6),
        )

        self.bottleneck_layer3 = nn.Sequential(
            BasicBlock(32, 64, 2, 6),
            BasicBlock(64, 64, 1, 6),
            BasicBlock(64, 64, 1, 6),
            BasicBlock(64, 64, 1, 6)
        )

        self.bottleneck_layer4 = nn.Sequential(
            BasicBlock(64, 96, 2, 6),
            BasicBlock(96, 96, 1, 6),
            BasicBlock(96, 96, 1, 6),
        )

        self.bottleneck_layer5 = nn.Sequential(
            BasicBlock(96, 160, 2, 6),
            BasicBlock(160, 160, 1, 6),
            BasicBlock(160, 160, 1, 6),
        )

        self.basic_block2 = BasicBlock(160, 320, 1, 6)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(1280)
        self.relu2 = nn.ReLU6()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, nb_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.basic_block1(x)
        x = self.bottleneck_layer1(x)
        x = self.bottleneck_layer2(x)
        x = self.bottleneck_layer3(x)
        x = self.bottleneck_layer4(x)
        x = self.bottleneck_layer5(x)
        x = self.basic_block2(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = MobileNetV2(10)
    model(torch.randn(1, 3, 224, 224))