import torch
from torch import nn
from torch.nn import functional as F


# originated from http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

class LeNet5(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120 * 24 * 24, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = self.pool(F.tanh(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    net = LeNet5(in_channels=3)
    net(torch.randn(1, 3, 224, 224))