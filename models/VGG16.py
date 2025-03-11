import torch
from torch import nn


# originated from https://arxiv.org/pdf/1409.1556

class ConvBlock(nn.Module):
    def __init__(self, channels):
        super(ConvBlock, self).__init__()
        self.conv_layer = nn.ModuleList(
            [nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1)
             for i in range(len(channels) - 1)]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.conv_layer)):
            x = self.conv_layer[i](x)
            x = self.relu(x)
        return x


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = ConvBlock([3, 64])
        self.conv2 = ConvBlock([64, 128])
        self.conv3 = ConvBlock([128, 256, 256])
        self.conv4 = ConvBlock([256, 512, 512])
        self.conv5 = ConvBlock([512, 512, 512])

        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.maxpool(self.conv3(x))
        x = self.maxpool(self.conv4(x))
        x = self.maxpool(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = VGG16()
    model(torch.randn(1, 3, 224, 224))