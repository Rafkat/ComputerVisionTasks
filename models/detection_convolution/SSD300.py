import math
import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from bs4 import BeautifulSoup
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision

from models.detection_convolution.SSD300_utils import cxcy_to_xy, decode_bboxes, find_IoU, distort, lighting_noise, \
    expand_filler, random_crop, random_flip

# originated from https://arxiv.org/pdf/1512.02325


classes_map = {'background': 0, 'person': 1, 'bird': 2, 'cat': 3, 'cow': 4, 'dog': 5, 'horse': 6, 'sheep': 7,
               'aeroplane': 8, 'bicycle': 9, 'boat': 10, 'bus': 11, 'car': 12, 'motorbike': 13, 'train': 14,
               'bottle': 15, 'chair': 16, 'diningtable': 17, 'pottedplant': 18, 'sofa': 19, 'tvmonitor': 20}


class TrainDataset(Dataset):
    def __init__(self, image_dir='./tasks/detection/fruits/data/images',
                 annot_dir='./tasks/detection/fruits/data/annotations', transform=None, image_size=300):
        self.transform = transform
        self.image_size = image_size

        self.image_dir = image_dir
        self.annot_dir = annot_dir

        self.image_paths = list(sorted(os.listdir(image_dir)))
        self.annot_paths = list(sorted(os.listdir(annot_dir)))

        self.mean = [0.485, 0.456, 0.406]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        random_value = random.random()
        image_path = self.image_paths[idx]
        annot_path = self.annot_paths[idx]
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        annot = open(os.path.join(self.annot_dir, annot_path)).read()
        soup = BeautifulSoup(annot, 'html.parser')
        classes = []
        bboxes = []
        difficulties = []
        img_width, img_height = int(soup.find('width').string), int(soup.find('height').string)
        for obj in soup.find_all('object'):
            classes.append(classes_map[obj.find('name').string])
            difficulties.append(int(obj.find('difficult').string))

            x_min = max(0, int(obj.find('xmin').string))
            y_min = max(0, int(obj.find('ymin').string))
            x_max = min(img_width, int(obj.find('xmax').string))
            y_max = min(img_height, int(obj.find('ymax').string))

            bbox = [x_min, y_min, x_max, y_max]
            if self.transform:
                x_min /= img_width
                x_max /= img_width
                y_min /= img_height
                y_max /= img_height
                bbox = [x_min, y_min, x_max, y_max]
            bboxes.append(bbox)

        bboxes = torch.FloatTensor(bboxes)
        classes = torch.LongTensor(classes)
        difficulties = torch.ByteTensor(difficulties)

        if self.transform:
            image = distort(image)
            image = lighting_noise(image)
            if random_value < 0.5:
                image, bboxes = expand_filler(image, bboxes, self.mean)

            image, bboxes, classes, difficulties = random_crop(image, bboxes, classes, difficulties)

            image, bboxes = random_flip(image, bboxes)

            image = self.transform(image)

        return image, bboxes, classes, difficulties


class SSDTrainDataLoader:
    def __init__(self, batch_size=128, random_seed=42, valid_size=0.2, shuffle=True):
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.valid_size = valid_size
        self.shuffle = shuffle

    @staticmethod
    def combine(batch):
        images = []
        boxes = []
        labels = []
        difficulties = []

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)
        return images, boxes, labels, difficulties

    def load_data(self, image_dir, annot_dir):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        valid_transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_dataset = TrainDataset(image_dir=image_dir, annot_dir=annot_dir, transform=train_transform)
        valid_dataset = TrainDataset(image_dir=image_dir, annot_dir=annot_dir, transform=valid_transform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.seed(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sample = SubsetRandomSampler(train_idx)
        valid_sample = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sample,
                                  collate_fn=self.combine, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, sampler=valid_sample,
                                  collate_fn=self.combine, num_workers=4, pin_memory=True)
        return train_loader, valid_loader


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


class VGG16Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16Encoder, self).__init__()
        self.conv1 = ConvBlock([3, 64, 64])
        self.conv2 = ConvBlock([64, 128, 128])
        self.conv3 = ConvBlock([128, 256, 256, 256])
        self.conv4 = ConvBlock([256, 512, 512, 512])
        self.conv5 = ConvBlock([512, 512, 512, 512])
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.maxpool3 = nn.MaxPool2d(3, 1, 1)

        if pretrained:
            self.load_pretrained()

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool1(self.conv2(x))
        x = self.maxpool2(self.conv3(x))
        out = self.conv4(x)
        x = self.maxpool1(out)
        x = self.maxpool3(self.conv5(x))
        return x, out

    def load_pretrained(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        for i, parameters in enumerate(param_names):
            state_dict[parameters] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict)


class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv8 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

    def forward(self, x, encoder_out):
        out_features = [encoder_out]
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        out_features.append(x)
        x = self.conv8(x)
        out_features.append(x)
        x = self.conv9(x)
        out_features.append(x)
        x = self.conv10(x)
        out_features.append(x)
        x = self.conv11(x)
        out_features.append(x)
        return out_features


class Head(nn.Module):
    def __init__(self, nb_classes=1000):
        super(Head, self).__init__()
        self.nb_classes = nb_classes
        self.loc_blocks = nn.ModuleList(
            [nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
             nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
             nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
             nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
             nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
             nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1), ]
        )

        self.class_blocks = nn.ModuleList(
            [nn.Conv2d(512, 4 * self.nb_classes, kernel_size=3, padding=1),
             nn.Conv2d(1024, 6 * self.nb_classes, kernel_size=3, padding=1),
             nn.Conv2d(512, 6 * self.nb_classes, kernel_size=3, padding=1),
             nn.Conv2d(256, 6 * self.nb_classes, kernel_size=3, padding=1),
             nn.Conv2d(256, 4 * self.nb_classes, kernel_size=3, padding=1),
             nn.Conv2d(256, 4 * self.nb_classes, kernel_size=3, padding=1), ]
        )

    def forward(self, out_features):
        b = out_features[-1].size(0)
        locs = []
        classes = []
        for feature, loc_block, class_block in zip(out_features, self.loc_blocks, self.class_blocks):
            loc = loc_block(feature).permute(0, 2, 3, 1).contiguous().view(b, -1, 4)
            class_feature = class_block(feature).permute(0, 2, 3, 1).contiguous().view(b, -1, self.nb_classes)
            locs.append(loc)
            classes.append(class_feature)

        locs_pred = torch.cat(locs, dim=1)
        classes_pred = torch.cat(classes, dim=1)
        return locs_pred, classes_pred


class SingleShotMultiBoxDetector(nn.Module):
    def __init__(self, nb_classes=1000):
        super(SingleShotMultiBoxDetector, self).__init__()
        self.encoder = VGG16Encoder()
        self.neck = Neck()
        self.head = Head(nb_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.default_boxes = self.create_default_boxes()
        self.nb_classes = nb_classes

    def forward(self, x):
        x, out = self.encoder(x)
        features = self.neck(x, out)
        locs_pred, class_pred = self.head(features)
        return locs_pred, class_pred

    def create_default_boxes(self):
        wh = [38, 19, 10, 5, 3, 1]
        scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
        aspect_ratios = [
            [1., 2., 0.5],
            [1., 2., 3., 0.5, 0.3333],
            [1., 2., 3., 0.5, 0.3333],
            [1., 2., 3., 0.5, 0.3333],
            [1., 2., 0.5],
            [1., 2., 0.5]
        ]

        default_boxes = []

        for k in range(len(wh)):
            for i in range(wh[k]):
                for j in range(wh[k]):
                    cx = (j + 0.5) / wh[k]
                    cy = (i + 0.5) / wh[k]

                    for ratio in aspect_ratios[k]:
                        default_boxes.append([cx, cy, scales[k] * math.sqrt(ratio),
                                              scales[k] / math.sqrt(ratio)])

                        if ratio == 1:
                            try:
                                add_scale = math.sqrt(scales[k] * scales[k + 1])
                            except IndexError:
                                add_scale = 1.
                            default_boxes.append([cx, cy, add_scale, add_scale])

        default_boxes = torch.FloatTensor(default_boxes).to(self.device)
        default_boxes.clamp_(0, 1)
        return default_boxes

    # Non-Maximum Suppression
    def detect(self, locs_pred, class_pred, min_score, max_overlap, top_k):
        batch_size = locs_pred.size(0)
        class_pred = F.softmax(class_pred, dim=2)

        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []

        for i in range(batch_size):
            decoded_locs = cxcy_to_xy(decode_bboxes(locs_pred[i], self.default_boxes))

            image_boxes = []
            image_labels = []
            image_scores = []

            for c in range(1, self.nb_classes):
                class_scores = class_pred[i][:, c]
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()

                if n_above_min_score == 0:
                    continue

                class_scores = class_scores[score_above_min_score]
                class_decoded_locs = decoded_locs[score_above_min_score]

                class_scores, sort_id = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_id]

                overlap = find_IoU(class_decoded_locs, class_decoded_locs)

                suppress = torch.zeros(n_above_min_score, dtype=torch.uint8).to(self.device)

                for box_id in range(class_decoded_locs.size(0)):
                    if suppress[box_id] == 1:
                        continue
                    condition = overlap[box_id] > max_overlap
                    condition = torch.tensor(condition, dtype=torch.uint8).to(self.device)
                    suppress = torch.max(suppress, condition)

                    suppress[box_id] = 0

                image_boxes.append(class_decoded_locs[~suppress.bool()])
                image_labels.append(torch.LongTensor((~suppress.bool()).sum().item() * [c]).to(self.device))
                image_scores.append(class_scores[~suppress.bool()])

            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(self.device))
                image_labels.append(torch.LongTensor([0]).to(self.device))
                image_scores.append(torch.FloatTensor([0.]).to(self.device))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            n_objects = image_scores.size(0)

            if n_objects > top_k:
                image_scores, sort_index = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]
                image_boxes = image_boxes[sort_index][:top_k]
                image_labels = image_labels[sort_index][:top_k]

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores


if __name__ == '__main__':
    net = VGG16Encoder()
    neck_net = Neck()
    head = Head()
    # print(net(torch.randn(1, 3, 300, 300))[1].shape)
    # print(neck_net(torch.randn(1, 512, 19, 19))[-1].shape)
    res = head(neck_net(torch.randn(1, 512, 19, 19), torch.randn(1, 512, 38, 38)))
    print(res[0].shape, res[1].shape)

    train_dataloader, val_dataloader = SSDTrainDataLoader(batch_size=16).load_data()
    print(len(train_dataloader))
    images, boxes, labels = next(iter(train_dataloader))
    # print(images.shape, boxes, labels)
