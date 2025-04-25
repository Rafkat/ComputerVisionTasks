import random

import PIL
import torch
import torchvision.transforms.functional as F


def intersect(boxes1, boxes2):
    n1 = boxes1.size(0)
    n2 = boxes2.size(0)
    max_xy = torch.min(boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2))

    min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2))
    inter = torch.clamp(max_xy - min_xy, min=0.)
    return inter[:, :, 0] * inter[:, :, 1]


def find_IoU(boxes1, boxes2):
    inter = intersect(boxes1, boxes2)
    area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    area_boxes1 = area_boxes1.unsqueeze(1).expand_as(inter)
    area_boxes2 = area_boxes2.unsqueeze(0).expand_as(inter)
    union = area_boxes1 + area_boxes2 - inter
    return inter / union


def xy_to_cxcy(bboxes):
    return torch.cat([(bboxes[:, 2:] + bboxes[:, :2]) / 2,
                      bboxes[:, 2:] - bboxes[:, :2]], 1)


def cxcy_to_xy(bboxes):
    return torch.cat([bboxes[:, :2] - (bboxes[:, 2:] / 2),
                      bboxes[:, :2] + (bboxes[:, 2:] / 2)], 1)


def encode_bboxes(bboxes, default_boxes):
    return torch.cat([(bboxes[:, :2] - default_boxes[:, :2]) / (default_boxes[:, 2:]),
                      torch.log(bboxes[:, 2:] / default_boxes[:, 2:])], 1)


def decode_bboxes(offsets, default_boxes):
    return torch.cat([offsets[:, :2] * default_boxes[:, 2:] + default_boxes[:, :2],
                      torch.exp(offsets[:, 2:]) * default_boxes[:, 2:]], 1)


def distort(image):
    distortions = [F.adjust_brightness,
                   F.adjust_contrast,
                   F.adjust_saturation]
    random.shuffle(distortions)

    for function in distortions:
        if random.random() < 0.5:
            adjust_factor = random.uniform(0.5, 1.5)
            image = function(image, adjust_factor)
    return image


def lighting_noise(image):
    if random.random() < 0.5:
        perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
        swap = perms[random.randint(0, len(perms) - 1)]
        image = F.to_tensor(image)
        image = image[swap, :, :]
        image = F.to_pil_image(image)
    return image


def expand_filler(image, bboxes, filler):
    if type(image) == PIL.Image.Image:
        image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    filler = torch.FloatTensor(filler)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h

    new_image[:, top:bottom, left:right] = image

    bboxes = bboxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)
    return image, bboxes


def random_crop(image, bboxes, classes, difficulties):
    if type(image) == PIL.Image.Image:
        image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)

    while True:
        mode = random.choice([0.1, 0.3, 0.5, 0.9, None])

        if mode is None:
            return image, bboxes, classes, difficulties

        new_image = image
        new_bboxes = bboxes
        new_difficulties = difficulties
        new_classes = classes

        for _ in range(50):
            new_h = random.uniform(0.3 * original_h, original_h)
            new_w = random.uniform(0.3 * original_w, original_w)

            if new_h / new_w < 0.5 or new_h / new_w > 2:
                continue

            left = random.uniform(0, original_w - new_w)
            right = left + new_w
            top = random.uniform(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([int(left), int(top), int(right), int(bottom)])

            overlap = find_IoU(crop.unsqueeze(0), bboxes)
            overlap = overlap.squeeze(0)
            if overlap.max().item() < mode:
                continue

            new_image = image[:, int(top):int(bottom), int(left):int(right)]

            center_bb = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0

            center_in_crop = ((center_bb[:, 0] > left) * (center_bb[:, 0] < right)
                              * (center_bb[:, 1] > top) * (center_bb[:, 1] < bottom))

            if not center_in_crop.any():
                continue

            new_bboxes = bboxes[center_in_crop, :]

            new_classes = classes[center_in_crop]

            new_difficulties = difficulties[center_in_crop]

            new_bboxes[:, :2] = torch.max(new_bboxes[:, :2], crop[:2])
            new_bboxes[:, :2] -= crop[:2]
            new_bboxes[:, 2:] = torch.min(new_bboxes[:, 2:], crop[2:])
            new_bboxes[:, 2:] -= crop[:2]

            return new_image, new_bboxes, new_classes, new_difficulties
        return new_image, new_bboxes, new_classes, new_difficulties


def random_flip(image, bboxes):
    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)

    if random.random() > 0.5:
        return image, bboxes

    new_image = F.hflip(image)

    bboxes[:, 0] = image.width - bboxes[:, 0]
    bboxes[:, 2] = image.width - bboxes[:, 2]
    bboxes = bboxes[:, [2, 1, 0, 3]]
    return new_image, bboxes
