import torch
import torch.nn as nn


# originated from https://arxiv.org/pdf/1610.02357

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, groups=in_channels, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.bn(x)


class EntryFlow(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EntryFlow, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.res_con1 = nn.Conv2d(64, 128, 1, stride=2)

        self.sep_conv1 = SeparableConv2d(64, 128)
        self.relu3 = nn.ReLU()

        self.sep_conv2 = SeparableConv2d(128, 128)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.res_con2 = nn.Conv2d(128, 256, 1, stride=2)

        self.relu4 = nn.ReLU()
        self.sep_conv3 = SeparableConv2d(128, 256)

        self.relu5 = nn.ReLU()
        self.sep_conv4 = SeparableConv2d(256, 256)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.res_con3 = nn.Conv2d(256, out_channels, 1, stride=2)

        self.relu6 = nn.ReLU()
        self.sep_conv5 = SeparableConv2d(256, out_channels)
        self.relu7 = nn.ReLU()
        self.sep_conv6 = SeparableConv2d(out_channels, out_channels)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        out1 = self.relu2(x)

        x = self.sep_conv1(out1)
        x = self.relu3(x)
        x = self.sep_conv2(x)
        out2 = self.res_con1(out1) + self.maxpool1(x)

        x = self.relu4(out2)
        x = self.sep_conv3(x)
        x = self.relu5(x)
        x = self.sep_conv4(x)
        out3 = self.res_con2(out2) + self.maxpool2(x)

        x = self.relu6(out3)
        x = self.sep_conv5(x)
        x = self.relu7(x)
        x = self.sep_conv6(x)
        out4 = self.res_con3(out3) + self.maxpool3(x)

        return out4


class MiddleFlow(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks):
        super(MiddleFlow, self).__init__()
        self.layer = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels, out_channels),
            nn.ReLU(),
            SeparableConv2d(out_channels, out_channels),
            nn.ReLU(),
            SeparableConv2d(out_channels, out_channels),
        )

        self.layers = nn.ModuleList([self.layer] * n_blocks)

    def forward(self, x):
        for layer in self.layers:
            input_tensor = x
            x = layer(x)
            x = input_tensor + x
        return x


class ExitFlow(nn.Module):
    def __init__(self, in_channels, nb_classes):
        super(ExitFlow, self).__init__()
        self.last_sep_layer = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels, in_channels),
            nn.ReLU(),
            SeparableConv2d(in_channels, 1024),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.res_con = nn.Conv2d(in_channels, 1024, kernel_size=1, stride=2)

        self.sep_conv1 = SeparableConv2d(1024, 1536)
        self.relu1 = nn.ReLU()
        self.sep_conv2 = SeparableConv2d(1536, 2048)
        self.relu2 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, nb_classes)

    def forward(self, x):
        input_tensor = x
        x = self.last_sep_layer(x)
        x = self.res_con(input_tensor) + x

        x = self.sep_conv1(x)
        x = self.relu1(x)
        x = self.sep_conv2(x)
        x = self.relu2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Xception(nn.Module):
    def __init__(self, intermediate_channels, nb_middle_blocks, nb_classes):
        super(Xception, self).__init__()
        self.entry_net = EntryFlow(3, intermediate_channels)
        self.middle_net = MiddleFlow(intermediate_channels, intermediate_channels, nb_middle_blocks)
        self.exit_net = ExitFlow(intermediate_channels, nb_classes)

    def forward(self, x):
        x = self.entry_net(x)
        x = self.middle_net(x)
        x = self.exit_net(x)
        return x


if __name__ == '__main__':
    model = Xception(728, 8, 10)
    model(torch.randn(1, 3, 224, 224))