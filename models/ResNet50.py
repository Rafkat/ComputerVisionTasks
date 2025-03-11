import torch
import torch.nn as nn


# originated from https://arxiv.org/pdf/1512.03385

class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, intermediate, out_channels, downsample=False):
        super(BottleneckResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, intermediate, kernel_size=1, stride=1)
        self.bn_sc = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(intermediate)
        if downsample:
            self.conv2 = nn.Conv2d(intermediate, intermediate, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        else:
            self.conv2 = nn.Conv2d(intermediate, intermediate, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Conv2d(intermediate, out_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(intermediate)
        self.conv3 = nn.Conv2d(intermediate, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if x.size() != out.size():
            shortcut = self.bn_sc(self.shortcut(x))
            return self.relu(out + shortcut)
        return self.relu(out + x)


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_layer1 = nn.Sequential(
            BottleneckResidualBlock(64, 64, 256),
            BottleneckResidualBlock(256, 64, 256),
            BottleneckResidualBlock(256, 64, 256),
        )

        self.res_layer2 = nn.Sequential(
            BottleneckResidualBlock(256, 128, 512, downsample=True),
            BottleneckResidualBlock(512, 128, 512),
            BottleneckResidualBlock(512, 128, 512),
            BottleneckResidualBlock(512, 128, 512),
        )

        self.res_layer3 = nn.Sequential(
            BottleneckResidualBlock(512, 256, 1024, downsample=True),
            BottleneckResidualBlock(1024, 256, 1024),
            BottleneckResidualBlock(1024, 256, 1024),
            BottleneckResidualBlock(1024, 256, 1024),
            BottleneckResidualBlock(1024, 256, 1024),
            BottleneckResidualBlock(1024, 256, 1024),
        )

        self.res_layer4 = nn.Sequential(
            BottleneckResidualBlock(1024, 512, 2048, downsample=True),
            BottleneckResidualBlock(2048, 512, 2048),
            BottleneckResidualBlock(2048, 512, 2048),
        )

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        x = self.res_layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = ResNet50()
    model(torch.randn(1, 3, 224, 224))