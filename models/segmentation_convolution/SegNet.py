import glob
import os
import random

import PIL
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor


# originated from https://arxiv.org/pdf/1511.00561

class DataLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path, img_size=(512, 512), transform=None):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path, 'Images', '*.jpg'))
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(
                os.path.join(folder_path, 'Labels', os.path.basename(img_path).split('.')[0] + '.png'))
        self.transform = transform
        self.to_tensor = transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
        ])

        if self.transform:
            self.angles = [0, 90, 180, 270]

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        image = Image.open(img_path)
        label = Image.open(mask_path)

        if self.transform:
            random_angle = random.choice(self.angles)
            image = TF.rotate(image, random_angle, interpolation=PIL.Image.BILINEAR, expand=True)
            label = TF.rotate(label, random_angle, interpolation=PIL.Image.BILINEAR, expand=True)

        image = self.to_tensor(image)
        label = self.to_tensor(label)

        label_bg = 1 - abs(abs(label[0, :, :] - label[1, :, :]) - label[2, :, :])
        label_bg = label_bg[None, :, :]
        label = torch.concat([label, label_bg], dim=0)

        return image, label

    def __len__(self):
        return len(self.img_files)


class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class DecoderForVGG16(torch.nn.Module):
    def __init__(self, channels=(512, 512, 256, 128, 64)):
        super().__init__()
        self.channels = channels
        self.upsample = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_blocks = torch.nn.ModuleList(
            [Block(channels[0], channels[0])] +
            [Block(channels[i], channels[i + 1]) for i in range(1, len(channels) - 1)]
        )

    def forward(self, x, ind, maxpool_out):
        for i in range(len(self.channels) - 1):
            x = self.upsample(x, ind, output_size=maxpool_out[i][0].size())
            ind = maxpool_out[i][1]
            x = self.dec_blocks[i](x)
        return x, ind


class VGG16Unet(torch.nn.Module):
    def __init__(self, dec_channels=(512, 512, 256, 128, 64), nb_classes=4, out_size=(512, 512)):
        super(VGG16Unet, self).__init__()
        self.vgg16_encoder = models.vgg16(weights='DEFAULT')
        self.decoder = DecoderForVGG16(dec_channels)
        self.conv = torch.nn.Conv2d(dec_channels[-1], dec_channels[-1], 3, padding=1)
        self.head = torch.nn.Conv2d(dec_channels[-1], nb_classes, kernel_size=1)
        self.out_size = out_size
        self.norm = torch.nn.BatchNorm2d(dec_channels[-1])
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.upsample = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.enc_outputs = create_feature_extractor(self.vgg16_encoder,
                                                    return_nodes={
                                                        'features.2': 'output64',
                                                        'features.7': 'output128',
                                                        'features.14': 'output256',
                                                        'features.21': 'output512',
                                                        'features.28': 'bottleneck'
                                                    })

    def forward(self, x):
        x = self.enc_outputs(x)
        enc_outputs = [x.get('output512'), x.get('output256'), x.get('output128'), x.get('output64')]
        maxpool_out = [self.maxpool(enc_out) for enc_out in enc_outputs]

        x = x.get('bottleneck')
        x, ind = self.maxpool(x)
        x, ind = self.decoder(x, ind, maxpool_out)
        x = self.upsample(x, ind)
        x = self.relu(self.norm(self.conv(x)))
        x = self.head(x)
        x = F.interpolate(x, self.out_size)
        return x


if __name__ == '__main__':
    model = VGG16Unet()
    model(torch.randn(1, 3, 512, 512))