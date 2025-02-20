import torch
from torch import nn


# originated from https://arxiv.org/pdf/1409.4842


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_proj):
        super(InceptionBlock, self).__init__()
        self.conv11 = nn.Conv2d(in_channels, n1x1, kernel_size=1)
        self.conv12 = nn.Conv2d(in_channels, n3x3red, kernel_size=1)
        self.conv13 = nn.Conv2d(in_channels, n5x5red, kernel_size=1)
        self.conv14 = nn.Conv2d(in_channels, pool_proj, kernel_size=1)

        self.conv31 = nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1)
        self.conv51 = nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv11(x))

        x2 = self.relu(self.conv12(x))
        x2 = self.relu(self.conv31(x2))

        x3 = self.relu(self.conv13(x))
        x3 = self.relu(self.conv51(x3))

        x4 = self.maxpool(x)
        x4 = self.relu(self.conv14(x4))

        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.Conv2d(64, 192, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.lrn = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool1 = nn.AvgPool2d(5, stride=3)
        self.out_conv1 = nn.Conv2d(512, 2048, kernel_size=1)
        self.out_fc11 = nn.Linear(2048, 1024)
        self.out_fc12 = nn.Linear(1024, 1000)
        self.avgpool2 = nn.AvgPool2d(5, stride=3)
        self.out_conv2 = nn.Conv2d(512, 2048, kernel_size=1)
        self.out_fc21 = nn.Linear(2048, 1024)
        self.out_fc22 = nn.Linear(1024, 1000)
        self.avgpool3 = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        outputs = []

        x = self.first_block(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        x = self.inception4a(x)

        out1 = self.out_conv1(self.avgpool1(x))
        outputs.append(self.out_fc12(self.out_fc11(out1.view(out1.size(0), -1))))

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        out2 = self.out_conv2(self.avgpool2(x))
        outputs.append(self.out_fc22(self.out_fc21(out2.view(out2.size(0), -1))))

        x = self.inception4e(x)
        x = self.maxpool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        outputs.append(x)
        return outputs


