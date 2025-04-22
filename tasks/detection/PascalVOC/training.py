from collections import defaultdict

import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models.detection_convolution.SSD import SSDTrainDataLoader
from tasks.utils import EarlyStopping, calculate_mAP, MultiBoxLoss


class PascalVOCTraining:
    def __init__(self, model, batch_size=16):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.train_dataloader, self.val_dataloader = SSDTrainDataLoader(batch_size=batch_size).load_data(
            './data/VOCdevkit/VOC2007/JPEGImages/',
            './data/VOCdevkit/VOC2007/Annotations/'
        )
        self.label_map = {'background': 0, 'person': 1, 'bird': 2, 'cat': 3, 'cow': 4, 'dog': 5, 'horse': 6, 'sheep': 7,
                          'aeroplane': 8, 'bicycle': 9, 'boat': 10, 'bus': 11, 'car': 12, 'motorbike': 13, 'train': 14,
                          'bottle': 15, 'chair': 16, 'diningtable': 17, 'pottedplant': 18, 'sofa': 19, 'tvmonitor': 20}
        self.rev_label_map = {v: k for k, v in self.label_map.items()}

    def one_epoch_train(self, optimizer, loss_func):
        self.model.train()
        total_loss = 0

        for i, (images, boxes, labels) in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            images = images.to(self.device)
            boxes = [b.to(self.device) for b in boxes]
            labels = [l.to(self.device) for l in labels]

            locs_pred, cls_pred = self.model(images)

            loss = loss_func(locs_pred, cls_pred, boxes, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss = total_loss / (len(self.train_dataloader) * self.batch_size)

        with torch.no_grad():
            self.model.eval()

            detect_boxes = []
            detect_labels = []
            detect_scores = []
            target_boxes = []
            target_labels = []
            for i, (images, boxes, labels) in enumerate(self.val_dataloader):
                images = images.to(self.device)
                boxes = [b.to(self.device) for b in boxes]
                labels = [l.to(self.device) for l in labels]

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
            APs, mAP = calculate_mAP(detect_boxes, detect_labels,
                                     detect_scores, target_boxes, target_labels,
                                     self.label_map, self.rev_label_map)

        return total_loss, APs, mAP

    def train(self, n_epochs, lr, wd, postfix):
        history = defaultdict(list)
        ssd_loss = MultiBoxLoss(self.model.default_boxes).to(self.device)
        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
        early_stopping = EarlyStopping(patience=10)

        for epoch in range(n_epochs):
            total_loss, APs, mAP = self.one_epoch_train(optimizer, ssd_loss)
            scheduler.step(total_loss)

            history['loss'].append(total_loss)
            history['APs'].append(APs)
            history['mAP'].append(mAP)

            print(f'[INFO]: Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss:.4f}, APs: {APs}, mAP: {mAP}')
            early_stopping(total_loss, self.model)
            if early_stopping.early_stop:
                print('Early stopping')
                break

        pd.DataFrame(history).to_csv(f'./tasks/detection/fruits/logs/ssd_history_{postfix}.csv', index=False)
        torch.save(self.model.state_dict(), f'./ssd_model_{postfix}.pth')
