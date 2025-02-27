import torch.nn as nn


# originated from https://arxiv.org/pdf/1611.05431v2


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()

        if downsample:
            self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        else:
            self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.bn2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(4, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x


class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, cardinality, downsample=False):
        super(ResLayer, self).__init__()
        self.layer = nn.ModuleList([
            ResBlock(in_channels, out_channels, downsample) for _ in range(cardinality)
        ])

        if downsample:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn_sc = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer[0](x)
        for layer in self.layer[1:]:
            out = out + layer(x)
        return out + self.relu(self.bn_sc(self.shortcut(x)))


class ResNeXt(nn.Module):
    def __init__(self, layers_config, cardinality=32):
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ResLayer(64, 256, cardinality),
            *nn.ModuleList([
                ResLayer(256, 256, cardinality) for _ in range(layers_config[0] - 1)
            ])
        )

        self.layer2 = nn.Sequential(
            ResLayer(256, 512, cardinality, downsample=True),
            *nn.ModuleList([
                ResLayer(512, 512, cardinality) for _ in range(layers_config[1] - 1)
            ])
        )

        self.layer3 = nn.Sequential(
            ResLayer(512, 1024, cardinality, downsample=True),
            *nn.ModuleList([
                ResLayer(1024, 1024, cardinality) for _ in range(layers_config[2] - 1)
            ])
        )

        self.layer4 = nn.Sequential(
            ResLayer(1024, 2048, cardinality, downsample=True),
            *nn.ModuleList([
                ResLayer(2048, 2048, cardinality) for _ in range(layers_config[3] - 1)
            ])
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, cardinality)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
