import torch

from models.detection_convolution.FasterRCNN import FasterRCNN, FasterRCNNTrainDataLoader
from models.detection_convolution.RCNN import RCNN, RCNNTrainDataLoader
from models.detection_convolution.SSD300 import SingleShotMultiBoxDetector, SSDTrainDataLoader
from tqdm import tqdm

from tasks.detection.utils import calculate_mAP

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
        self.label_map = {'background': 0, 'pineapple': 1, 'snake fruit': 2, 'dragon fruit': 3, 'banana': 4}
        self.rev_label_map = {v: k for k, v in self.label_map.items()}

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
            target_boxes = []
            target_labels = []
            for i, (images, boxes, labels, difficulties) in tqdm(enumerate(self.val_dataloader),
                                                                 total=len(self.val_dataloader)):
                images = images.to(device)
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]
                difficulties = [d.to(device) for d in difficulties]

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
                                     detect_scores, target_boxes, target_labels, difficulties,
                                     self.label_map, self.rev_label_map)

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


class FasterRCNNTraining:
    def __init__(self, batch_size=16,
                 image_dir='./tasks/detection/fruits/data/images',
                 annot_dir='./tasks/detection/fruits/data/annotations'):
        self.batch_size = batch_size
        self.classes_map = {'background': 0, 'pineapple': 1, 'snake fruit': 2, 'dragon fruit': 3, 'banana': 4}
        self.model = FasterRCNN(n_classes=len(self.classes_map))
        self.train_dataloader, self.val_dataloader = FasterRCNNTrainDataLoader(classes_map=self.classes_map,
                                                                               batch_size=self.batch_size).load_data(
            image_dir, annot_dir)

    def train(self, optimizer, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for (images, boxes, labels, _) in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
                images = images.to(device)
                loss = self.model(images, boxes, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            total_loss = total_loss / len(self.train_dataloader)
            print(f'[INFO] Epoch {epoch + 1} / {epochs} | Loss: {total_loss:.4f}')
        torch.save(self.model.state_dict(), './faster_rcnn.pth')


if __name__ == '__main__':
    # ssd_train = SSDTraining()
    # ssd_loss = MultiBoxLoss(ssd_train.default_boxes)
    # ssd_train.model.to(device)
    # ssd_train.train(ssd_loss, optimizer=torch.optim.Adam(ssd_train.model.parameters(), lr=0.001))

    frcnn_train = FasterRCNNTraining(batch_size=4)
    optimizer = torch.optim.Adam(frcnn_train.model.parameters(), lr=0.001)
    epochs = 1000
    frcnn_train.train(optimizer, epochs)
