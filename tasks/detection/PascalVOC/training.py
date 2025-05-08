from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm

from models.detection_convolution.SSD300 import SSDTrainDataLoader, SingleShotMultiBoxDetector
from tasks.detection.utils import calculate_mAP, MultiBoxLoss


class PascalVOCTraining:
    def __init__(self, model, batch_size=16,
                 img_dir='./data/VOCdevkit/VOC2007/JPEGImages/',
                 annot_dir='./data/VOCdevkit/VOC2007/Annotations/'):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.train_dataloader, self.val_dataloader = SSDTrainDataLoader(batch_size=batch_size).load_data(
            image_dir=img_dir, annot_dir=annot_dir)
        self.label_map = {'background': 0, 'person': 1, 'bird': 2, 'cat': 3, 'cow': 4, 'dog': 5, 'horse': 6, 'sheep': 7,
                          'aeroplane': 8, 'bicycle': 9, 'boat': 10, 'bus': 11, 'car': 12, 'motorbike': 13, 'train': 14,
                          'bottle': 15, 'chair': 16, 'diningtable': 17, 'pottedplant': 18, 'sofa': 19, 'tvmonitor': 20}
        self.rev_label_map = {v: k for k, v in self.label_map.items()}
        self.iterations = 145000
        self.decay_lr_at = [96500, 120000]
        self.decay_lr_to = 0.1

    def one_epoch_train(self, optimizer, loss_func):
        self.model.train()
        total_loss = 0
        total_loc_loss = 0
        total_cls_loss = 0

        for i, (images, boxes, labels, _) in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            images = images.to(self.device)
            boxes = [box.to(self.device) for box in boxes]
            labels = [label.to(self.device) for label in labels]

            locs_pred, cls_pred = self.model(images)

            loc_loss, confidence_loss = loss_func(locs_pred, cls_pred, boxes, labels)
            loss = loc_loss + confidence_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loc_loss += loc_loss.item()
            total_cls_loss += confidence_loss.item()

        total_loss = total_loss / len(self.train_dataloader)
        total_loc_loss = total_loc_loss / len(self.train_dataloader)
        total_cls_loss = total_cls_loss / len(self.train_dataloader)
        return total_loss, total_loc_loss, total_cls_loss

    def train(self, lr, postfix):
        n_epochs = self.iterations // len(self.train_dataloader)
        decay_lr_at = [it // len(self.train_dataloader) for it in self.decay_lr_at]
        history = defaultdict(list)
        ssd_loss = MultiBoxLoss(self.model.default_boxes).to(self.device)
        self.model.to(self.device)

        biases, weights = [], []
        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in param_name:
                    biases.append(param)
                else:
                    weights.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': weights}], lr=lr, momentum=0.9,
                                    weight_decay=5e-4)

        for epoch in range(n_epochs):
            if epoch in decay_lr_at:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * self.decay_lr_to
            total_loss, total_loc_loss, total_cls_loss = self.one_epoch_train(optimizer, ssd_loss)

            history['loss'].append(total_loss)

            print(
                f'[INFO]: Epoch {epoch + 1}/{n_epochs}, Total Loss: {total_loss:.4f}, Loc Loss: {total_loc_loss:.4f}, '
                f'Conf Loss: {total_cls_loss:.4f}')

        pd.DataFrame(history).to_csv(f'./tasks/detection/PascalVOC/logs/ssd_history_{postfix}.csv', index=False)
        torch.save(self.model.state_dict(), f'./ssd_model_{postfix}.pth')

    def eval(self):
        with torch.no_grad():
            self.model.eval()

            detect_boxes = []
            detect_labels = []
            detect_scores = []
            target_boxes = []
            target_labels = []
            target_difficulties = []
            for i, (images, boxes, labels, difficulties) in tqdm(enumerate(self.val_dataloader),
                                                                 total=len(self.val_dataloader)):
                images = images.to(self.device)
                boxes = [b.to(self.device) for b in boxes]
                labels = [l.to(self.device) for l in labels]
                difficulties = [d.to(self.device) for d in difficulties]

                locs_pred, cls_pred = self.model(images)
                detect_boxes_batch, detect_labels_batch, detect_score_batch = self.model.detect(locs_pred, cls_pred,
                                                                                                min_score=0.01,
                                                                                                max_overlap=0.45,
                                                                                                top_k=200)
                detect_boxes.extend(detect_boxes_batch)
                detect_labels.extend(detect_labels_batch)
                detect_scores.extend(detect_score_batch)
                target_boxes.extend(boxes)
                target_labels.extend(labels)
                target_difficulties.extend(difficulties)
            APs, mAP = calculate_mAP(detect_boxes, detect_labels, detect_scores,
                                     target_boxes, target_labels, target_difficulties,
                                     self.label_map, self.rev_label_map)

        return APs, mAP


if __name__ == '__main__':
    model = SingleShotMultiBoxDetector(nb_classes=21).to(device='cuda')
    model.load_state_dict(torch.load('../../../ssd_model__sgd_pure.pth', weights_only=True))
    evaluation = PascalVOCTraining(model, batch_size=16, img_dir='../../../data/VOCdevkit/VOC2007/JPEGImages/',
                                   annot_dir='../../../data/VOCdevkit/VOC2007/Annotations/')
    APs, mAP = evaluation.eval()
    print(f'Average precision: {APs}, mean Average Precision: {mAP}')
