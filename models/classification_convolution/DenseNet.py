import torch
import torch.nn as nn


# originated from https://arxiv.org/pdf/1608.06993v5

class BasicBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1)

        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, block_num):
        super(DenseBlock, self).__init__()
        self.first_block = BasicBlock(in_channels, growth_rate)
        self.block = nn.ModuleList([BasicBlock(n * growth_rate, growth_rate) for n in range(1, block_num)])

    def forward(self, x):
        outputs = []
        x = self.first_block(x)
        outputs.append(x)
        for basic_block in self.block:
            out = basic_block(x)
            x = torch.cat(outputs + [out], dim=1)
            outputs.append(out)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(TransitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(self.relu1(self.bn1(x)))
        x = self.avgpool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, block_config=(6, 12, 32, 32), growth_rate=12, nb_classes=1000):
        super(DenseNet, self).__init__()
        self.dense_block1 = DenseBlock(growth_rate * 2, growth_rate, block_config[0])
        self.dense_block2 = DenseBlock(growth_rate * 4, growth_rate, block_config[1])
        self.dense_block3 = DenseBlock(growth_rate * 4, growth_rate, block_config[2])
        self.dense_block4 = DenseBlock(growth_rate * 4, growth_rate, block_config[3])

        self.transition1 = TransitionLayer(in_channels=block_config[0] * growth_rate, growth_rate=growth_rate)
        self.transition2 = TransitionLayer(in_channels=block_config[1] * growth_rate, growth_rate=growth_rate)
        self.transition3 = TransitionLayer(in_channels=block_config[2] * growth_rate, growth_rate=growth_rate)

        self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=growth_rate * 2, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(block_config[2] * growth_rate, nb_classes)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        x = self.maxpool(x)
        x = self.dense_block1(x)
        x = self.transition1(x)
        x = self.dense_block2(x)
        x = self.transition2(x)
        x = self.dense_block3(x)
        x = self.transition3(x)
        x = self.dense_block4(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = DenseNet(block_config=[6, 12, 32, 32], growth_rate=12)  # DenseNet169 config
    model(torch.randn(1, 3, 224, 224))
