import torch

from models.detection_convolution.RCNN import RCNN, RCNNTrainDataLoader
from models.detection_convolution.SSD import SingleShotMultiBoxDetector, SSDTrainDataLoader
from tasks.detection.fruits.utils import calculate_mAP, MultiBoxLoss
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SSDTraining:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.model = SingleShotMultiBoxDetector(nb_classes=5)
        self.default_boxes = self.model.create_default_boxes()
        self.train_dataloader, self.val_dataloader = SSDTrainDataLoader(
            batch_size=self.batch_size).load_data(
            './tasks/detection/fruits/data/images',
            './tasks/detection/fruits/data/annotations')

    def train(self, criterion, optimizer):
        self.model.train()
        total_loss = 0

        for i, (images, boxes, labels) in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            locs_pred, cls_pred = self.model(images)

            loss = criterion(locs_pred, cls_pred, boxes, labels)

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
            t_boxes = []
            t_labels = []
            for i, (images, boxes, labels) in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader)):
                images = images.to(device)
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]

                locs_pred, cls_pred = self.model(images)
                detect_boxes_batch, detect_labels_batch, detect_score_batch = self.model.detect(locs_pred, cls_pred,
                                                                                                min_score=0.01,
                                                                                                max_overlap=0.45,
                                                                                                top_k=200)
                detect_boxes.extend(detect_boxes_batch)
                detect_labels.extend(detect_labels_batch)
                detect_scores.extend(detect_score_batch)
                t_boxes.extend(boxes)
                t_labels.extend(labels)
            APs, mAP = calculate_mAP(detect_boxes, detect_labels, detect_scores, t_boxes, t_labels)

        return total_loss, APs, mAP


class RCNNTraining:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.rcnn = RCNN()
        self.model = self.rcnn.classifier
        self.train_dataloader, self.val_dataloader = RCNNTrainDataLoader(batch_size=self.batch_size).load_data()

    def train(self, criterion, optimizer, epochs):
        history = self.rcnn.train(self.train_dataloader, self.val_dataloader, criterion, optimizer, epochs)
        return history


if __name__ == '__main__':
    ssd_train = SSDTraining()
    ssd_loss = MultiBoxLoss(ssd_train.default_boxes)
    ssd_train.model.to(device)
    ssd_train.train(ssd_loss, optimizer=torch.optim.Adam(ssd_train.model.parameters(), lr=0.001))
