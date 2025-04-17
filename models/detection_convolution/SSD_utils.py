import torch


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
    return torch.cat([(bboxes[:, :2] - default_boxes[:, :2]) / (default_boxes[:, 2:] / 10),
                      torch.log(bboxes[:, 2:] / default_boxes[:, 2:]) * 5], 1)


def decode_bboxes(offsets, default_boxes):
    return torch.cat([offsets[:, :2] * default_boxes[:, 2:] / 10 + default_boxes[:, :2],
                      torch.exp(offsets[:, 2:] / 5) * default_boxes[:, 2:]], 1)
