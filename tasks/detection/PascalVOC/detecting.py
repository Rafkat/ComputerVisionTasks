import random

import torch

from models.detection_convolution.SSD300 import SingleShotMultiBoxDetector
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import matplotlib.pyplot as plt


class PascalVOCDetecting:
    def __init__(self, model, nb_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.label_map = {'background': 0, 'person': 1, 'bird': 2, 'cat': 3, 'cow': 4, 'dog': 5, 'horse': 6, 'sheep': 7,
                          'aeroplane': 8, 'bicycle': 9, 'boat': 10, 'bus': 11, 'car': 12, 'motorbike': 13, 'train': 14,
                          'bottle': 15, 'chair': 16, 'diningtable': 17, 'pottedplant': 18, 'sofa': 19, 'tvmonitor': 20}
        self.rev_label_map = {v: k for k, v in self.label_map.items()}

        self.distinct_colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                                for i in range(nb_classes)]
        self.label_color_map = {k: self.distinct_colors[i] for i, k in enumerate(self.label_map.keys())}

    def detect_objects(self, original_image, suppress=None):
        resize = transforms.Resize((300, 300))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        image = normalize(to_tensor(resize(original_image)))

        image = image.to(self.device)

        locs_pred, cls_pred = self.model(image.unsqueeze(0))

        detect_boxes, detect_labels, detect_scores = self.model.detect(locs_pred, cls_pred,
                                                                       min_score=0.4, max_overlap=0.45,
                                                                       top_k=200)

        detect_boxes = detect_boxes[0].to('cpu')
        original_dims = torch.FloatTensor(
            [original_image.width, original_image.height, original_image.width,
             original_image.height]).unsqueeze(0)

        detect_boxes = detect_boxes * original_dims

        detect_labels = [self.rev_label_map[l] for l in detect_labels[0].to('cpu').tolist()]

        if detect_labels == ['background']:
            return original_image

        annotated_image = original_image
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.truetype('arial.ttf', 15)

        for i in range(detect_boxes.size(0)):
            if suppress is not None:
                if detect_labels[i] in suppress:
                    continue

            box_location = detect_boxes[i].tolist()
            draw.rectangle(xy=box_location, outline=self.label_color_map[detect_labels[i]])
            draw.rectangle(xy=[l + 1. for l in box_location], outline=self.label_color_map[detect_labels[i]])

            text_size = font.getbbox(detect_labels[i].upper())
            text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
            textbox_location = [box_location[0], box_location[1] - text_size[1],
                                box_location[0] + text_size[0] + 4., box_location[1]]
            draw.rectangle(xy=textbox_location, fill=self.label_color_map[detect_labels[i]])
            draw.text(xy=text_location, text=detect_labels[i].upper(), fill='white', font=font)

        return annotated_image


if __name__ == '__main__':
    model = SingleShotMultiBoxDetector(nb_classes=21)
    model.load_state_dict(torch.load('../../../ssd_model__sgd_pure.pth', weights_only=True))
    detection = PascalVOCDetecting(model, nb_classes=21)
    original_image = Image.open('../../../data/VOCdevkit/VOC2007/JPEGImages/000012.jpg')
    original_image = original_image.convert('RGB')
    annotated_image = detection.detect_objects(original_image)
    plt.imshow(annotated_image)
    plt.show()
