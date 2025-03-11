import torch
from torch import nn
from torch.nn import functional as F


# originated from https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Linear(256 * 6 * 6, 4096)  # hard code that will now allow to use diverse image sizes
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1000)

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 3), stride=2)

    def forward(self, x):
        x = self.maxpool(self.bn1(F.relu(self.conv1(x))))
        x = self.maxpool(self.bn2(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(F.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x

if __name__ == '__main__':
    net = AlexNet()
    net(torch.randn(1, 3, 224, 224))