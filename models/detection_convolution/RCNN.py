import os
from collections import defaultdict

import numpy as np
import torch
import torchvision.models
from tqdm import tqdm
from bs4 import BeautifulSoup
import cv2
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tasks.utils import EarlyStopping


# originated from https://arxiv.org/pdf/1311.2524

class Config:
    def __init__(self):
        # Config
        self.NUM_CLASSES = 5
        self.CLASSES = ['nothing', 'pineapple', 'snake fruit', 'dragon fruit', 'banana']
        self.COLORS = ['black', 'red', 'green', 'blue', 'yellow']

        self.IMG_FOLDER = "./images"
        self.ANNOT_FOLDER = "./annotations"

        self.MAX_PROPOSED_ROI = 2000

        self.LIMIT_POS_PROP_ROI = 40
        self.LIMIT_NEG_PROP_ROI = 10

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.IMG_SIZE = 384
        self.BATCH_SIZE = 16
        self.NUM_EPOCHS = 100
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 0.0001

        self.MAX_PROP_INFER_ROI = 1000

        self.NMS_THRESHOLD = 0.1
        self.CLASSIFIER_PRETRAINED_PATH = "./rcnn_model.pth"


class TrainDataset(Dataset):
    def __init__(self, data_folder='./tasks/detection/fruits/data/boxes_v2',
                 dataset_file='./tasks/detection/fruits/data/boxes_v2.txt',
                 transform=None):
        self.transform = transform

        f = open(dataset_file, 'r')
        self.class_index_list = []
        self.image_path_list = []
        for line in f.readlines():
            class_index, file_name = [x.strip() for x in line.split(' ')]
            image_path = os.path.join(data_folder, file_name)
            self.class_index_list.append(int(class_index))
            self.image_path_list.append(image_path)
        f.close()

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        label = self.class_index_list[idx]
        image_path = self.image_path_list[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        return image, label


class RCNNTrainDataLoader:
    def __init__(self, batch_size=128, random_seed=42, valid_size=0.2, shuffle=True):
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.valid_size = valid_size
        self.shuffle = shuffle

    def load_data(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        valid_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_dataset = TrainDataset(transform=train_transform)
        valid_dataset = TrainDataset(transform=valid_transform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.seed(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, sampler=valid_sampler)

        return train_loader, valid_loader


class InferDataset(Dataset):
    def __init__(self, prop_array=None, infer_transform=None):
        self.prop_array = prop_array
        self.infer_transform = infer_transform

    def __len__(self):
        return len(self.prop_array)

    def __getitem__(self, idx):
        prop_roi = self.prop_array[idx]
        prop_roi = cv2.cvtColor(prop_roi, cv2.COLOR_BGR2RGB)
        pil_roi = Image.fromarray(prop_roi)
        if self.infer_transform:
            pil_roi = self.infer_transform(pil_roi)
        return pil_roi


class RegionProposalExtraction:
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    def __init__(self, image_name_path, annot_name_path):
        self.image_name_path = image_name_path
        self.annot_name_path = annot_name_path
        self.image_name_list = list(sorted(os.listdir(image_name_path)))
        self.annot_name_list = list(sorted(os.listdir(annot_name_path)))
        self.total_boxes = 0
        self.config = Config()
        self.save_box_path = '../../tasks/detection/fruits/data/boxes_v2'
        self.save_text_path = open('../../tasks/detection/fruits/data/boxes_v2.txt', 'a')

    def get_roi(self):
        for i, annot_name in tqdm(enumerate(self.annot_name_list), total=len(self.annot_name_list)):
            full_annot_path = os.path.join(self.annot_name_path, annot_name)
            full_image_path = os.path.join(self.image_name_path, self.image_name_list[i])

            contents_xml = open(full_annot_path).read()
            soup = BeautifulSoup(contents_xml, 'html.parser')

            img_width = int(soup.find('width').string)
            img_height = int(soup.find('height').string)

            ground_truth_boxes, ground_truth_class_indices = self.get_ground_truth_boxes_classes(soup,
                                                                                                 img_width,
                                                                                                 img_height)

            image, rects = self.get_image_prop_boxes(full_image_path)
            proposed_rects = [(x_min, y_min, x_min + w, y_min + h) for (x_min, y_min, w, h) in rects]

            self.get_gt_positive_samples(ground_truth_boxes, ground_truth_class_indices, image)
            self.get_prop_boxes(proposed_rects, ground_truth_boxes, ground_truth_class_indices, image)

        self.save_text_path.close()

    @classmethod
    def get_image_prop_boxes(cls, image_path):
        image = cv2.imread(image_path)
        cls.ss.setBaseImage(image)
        cls.ss.switchToSelectiveSearchFast()
        return image, cls.ss.process()

    def get_ground_truth_boxes_classes(self, soup, img_width, img_height):
        ground_truth_class_indices = []
        ground_truth_boxes = []
        for obj in soup.find_all('object'):
            # get ground-truth class
            class_name = obj.find('name').string
            class_index = self.config.CLASSES.index(class_name)
            ground_truth_class_indices.append(class_index)

            # get ground-truth box
            x_min = int(obj.find('xmin').string)
            x_max = int(obj.find('xmax').string)
            y_min = int(obj.find('ymin').string)
            y_max = int(obj.find('ymax').string)

            # truncate boxes, exceeding image sizes
            x_min = max(0, x_min)
            x_max = min(img_width, x_max)
            y_min = max(0, y_min)
            y_max = min(img_height, y_max)

            ground_truth_boxes.append([x_min, y_min, x_max, y_max])

        return ground_truth_boxes, ground_truth_class_indices

    @staticmethod
    def compute_iou(box_a, box_b):
        """
          input:
            - box_a: [x_a_min, y_a_min, x_a_max, y_a_max]
            - box_b: [x_b_min, y_b_min, x_b_max, y_b_max]
        """
        x_a_min, y_a_min, x_a_max, y_a_max = box_a
        x_b_min, y_b_min, x_b_max, y_b_max = box_b

        # find (x, y) coordinates of intersection box
        x_inters_min = max(x_a_min, x_b_min)
        y_inters_min = max(y_a_min, y_b_min)
        x_inters_max = min(x_a_max, x_b_max)
        y_inters_max = min(y_a_max, y_b_max)

        # compute area of intersection
        inters_area = max(0, x_inters_max - x_inters_min + 1) * max(0, y_inters_max - y_inters_min + 1)
        # area = 0 in case 2 boxes don't have any intersection

        # compute area of union = (box_a_area + box_b_area - inters_area)
        union_area = (x_a_max - x_a_min + 1) * (y_a_max - y_a_min + 1) + (x_b_max - x_b_min + 1) * (
                y_b_max - y_b_min + 1) - inters_area

        # compute iou = inters_area / union_area
        iou = inters_area / union_area

        return iou

    @staticmethod
    def is_overlapping(box_a, box_b):
        x_a_min, y_a_min, x_a_max, y_a_max = box_a
        x_b_min, y_b_min, x_b_max, y_b_max = box_b

        full_overlap = x_a_min > x_b_min and y_a_min >= y_b_min and x_a_max <= x_b_max and y_a_max <= y_b_max
        full_overlap2 = x_b_min >= x_a_min and y_b_min >= y_a_min and x_b_max <= x_a_max and y_b_max <= y_a_max

        if not full_overlap and full_overlap2:
            return False
        return True

    def get_gt_positive_samples(self, ground_truth_boxes, ground_truth_class_indices, image):
        # added ground truth images as positive samples
        for j, gt_box in enumerate(ground_truth_boxes):
            class_index_roi = ground_truth_class_indices[j]

            gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_box
            gt_roi = image[gt_y_min:gt_y_max, gt_x_min:gt_x_max]

            if gt_roi.size <= 0:
                continue

            gt_box_name = f'{self.total_boxes}.jpg'
            gt_box_path = os.path.join(self.save_box_path, gt_box_name)
            self.total_boxes += 1

            cv2.imwrite(gt_box_path, gt_roi)

            self.save_text_path.write(f'{class_index_roi} {gt_box_name}\n')

    def get_prop_boxes(self, proposed_rects, ground_truth_boxes, ground_truth_class_indices, image):
        n_pos = 0
        n_neg = 0
        # get proposed boxes
        for prop_rect in proposed_rects[:min(len(proposed_rects), self.config.MAX_PROPOSED_ROI)]:
            prop_x_min, prop_y_min, prop_x_max, prop_y_max = prop_rect

            prop_box, prop_box_name, prop_box_path, class_index, chk_neg = self.is_negative(ground_truth_boxes,
                                                                                            ground_truth_class_indices,
                                                                                            prop_rect, image, n_pos,
                                                                                            n_neg)

            if chk_neg:
                prop_box = image[prop_y_min:prop_y_max, prop_x_min:prop_x_max]

                prop_box_name = f'{self.total_boxes}.jpg'
                prop_box_path = os.path.join(self.save_box_path, prop_box_name)

                class_index = 0

            if (prop_box is not None
                    and prop_box_name is not None
                    and prop_box_path is not None
                    and class_index is not None):

                self.total_boxes += 1
                cv2.imwrite(prop_box_path, prop_box)
                self.save_text_path.write(f'{class_index} {prop_box_name}\n')

                if class_index == 0:
                    n_neg += 1
                else:
                    n_pos += 1

    def is_negative(self, ground_truth_boxes, ground_truth_class_indices, prop_rect, image, n_pos, n_neg):
        prop_box = prop_box_name = prop_box_path = class_index = None
        chk_neg = False
        prop_x_min, prop_y_min, prop_x_max, prop_y_max = prop_rect
        area_prop_rect = (prop_x_max - prop_x_min + 1) * (prop_y_max - prop_y_min + 1)
        for j, gt_box in enumerate(ground_truth_boxes):
            gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_box
            class_index_roi = ground_truth_class_indices[j]
            iou = self.compute_iou(gt_box, prop_rect)

            if iou >= 0.6 and n_pos < self.config.LIMIT_POS_PROP_ROI:
                prop_box = image[prop_y_min:prop_y_max, prop_x_min:prop_x_max]

                prop_box_name = f'{self.total_boxes}.jpg'
                prop_box_path = os.path.join(self.save_box_path, prop_box_name)

                class_index = class_index_roi
                break

            full_overlap = (prop_x_min >= gt_x_min and prop_y_min >= gt_y_min
                            and prop_x_max <= gt_x_max and prop_y_max <= gt_y_max)

            full_overlap_2 = (gt_x_min >= prop_x_min and gt_y_min >= prop_y_min
                              and gt_x_max <= prop_x_max and gt_y_max <= prop_y_max)

            if (not full_overlap
                    and not full_overlap_2
                    and iou <= 0.05
                    and n_neg < self.config.LIMIT_POS_PROP_ROI
                    and area_prop_rect >= 1000):
                chk_neg = True

        return prop_box, prop_box_name, prop_box_path, class_index, chk_neg


class RCNN:
    def __init__(self, pretrained=False):
        super(RCNN, self).__init__()
        self.config = Config()
        self.classifier = torchvision.models.efficientnet_v2_s(weights='DEFAULT')
        self.classifier.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=1280, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=self.config.NUM_CLASSES),
        )
        if pretrained:
            self.classifier.load_state_dict(torch.load(self.config.CLASSIFIER_PRETRAINED_PATH))

        self.prop_rects_extractor = RegionProposalExtraction.get_image_prop_boxes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier.to(self.device)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.infer_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def train(self, train_dataloader, val_dataloader, criterion, optimizer, epochs):
        history = defaultdict(list)
        self.classifier.to(self.device)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
        early_stopping = EarlyStopping(patience=10)
        for e in range(epochs):
            self.classifier.train()
            size = 0
            train_loss = 0
            train_acc = 0
            for images, labels in tqdm(train_dataloader, total=len(train_dataloader)):
                images = images.to(self.device)
                labels = labels.to(self.device)

                size += labels.size(0)

                outputs = self.classifier(images)

                loss = criterion(outputs, labels)
                train_loss += loss.item()

                y_hat = F.softmax(outputs, dim=1)
                value_softmax, index_softmax = torch.max(y_hat.data, dim=1)
                train_acc += (index_softmax == labels).sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss /= len(train_dataloader)
            train_acc /= size

            with torch.no_grad():
                size = 0
                test_loss = 0
                test_acc = 0
                self.classifier.eval()
                for images, labels in val_dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    size += labels.size(0)

                    outputs = self.classifier(images)

                    loss = criterion(outputs, labels)
                    test_loss += loss.item()

                    y_hat = F.softmax(outputs, dim=1)
                    value_softmax, index_softmax = torch.max(y_hat.data, dim=1)
                    test_acc += (index_softmax == labels).sum().item()

            test_loss /= len(val_dataloader)
            test_acc /= size
            scheduler.step(test_loss)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            print(f"[INFO]: Epoch {e + 1} / {epochs}, Train loss: {train_loss:.4f}, "
                  f"Train acc: {train_acc:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

            early_stopping(test_loss, self.classifier)
            if early_stopping.early_stop:
                print('Early stopping')
                break

        return history

    def detect(self, image_path):
        image, prop_rects = self.prop_rects_extractor(image_path)

        proposals = []
        rois = []
        for prop_rect in prop_rects[:min(len(prop_rects), self.config.MAX_PROP_INFER_ROI)]:
            prop_x_min, prop_y_min, prop_w, prop_h = prop_rect
            prop_x_max, prop_y_max = prop_x_min + prop_h, prop_y_min + prop_h
            roi = image[prop_y_min:prop_y_max, prop_x_min:prop_x_max]

            proposals.append(roi)
            rois.append((prop_x_min, prop_y_min, prop_x_max, prop_y_max))

        rois = np.asanyarray(rois)

        infer_dataset = InferDataset(prop_array=proposals, infer_transform=self.infer_transform)
        infer_dataloader = DataLoader(dataset=infer_dataset, batch_size=8, shuffle=False)

        probabilities = []
        for roi in tqdm(infer_dataloader, total=len(infer_dataloader)):
            roi = roi.to(self.device)

            outputs = self.classifier(roi)
            prob = F.softmax(outputs, dim=1)

            prob_np = prob.detach().cpu().numpy()

            for pr in prob_np:
                probabilities.append(pr)

        probabilities = np.asarray(probabilities)
        value_softmax, index_softmax = torch.max(torch.tensor(probabilities), dim=1)

        index_softmax_arr = np.array(index_softmax)
        value_softmax_arr = np.array(value_softmax)

        indices = np.where((index_softmax_arr != 0) & (value_softmax_arr > 0.95))[0]

        value_softmax_detec = value_softmax_arr[indices]
        index_softmax_detec = index_softmax_arr[indices]
        prop_obj_detec = probabilities[indices]
        rois_obj_detec = rois[indices]

        unique_class = np.unique(index_softmax_detec)

        P = np.arange(0, len(rois_obj_detec))
        keep = []

        counter = 0
        while len(P) > 0:
            counter += 1
            for idx_cl in unique_class:
                idx_sm_box_cl = np.where(index_softmax_detec == idx_cl)[0]
                idx_sm_box_cl = np.intersect1d(idx_sm_box_cl, P)

                if len(idx_sm_box_cl) == 0:
                    continue

                val_sm_box_cl = value_softmax_detec[idx_sm_box_cl]
                box_cl = rois_obj_detec[idx_sm_box_cl]
                sorted_val_sm_box_cl = np.argsort(val_sm_box_cl)
                max_box_cl = box_cl[sorted_val_sm_box_cl[-1]]
                max_idx_sm_box_cl = idx_sm_box_cl[sorted_val_sm_box_cl[-1]]

                P = np.delete(P, P == max_idx_sm_box_cl)
                keep.append(max_idx_sm_box_cl)

                for i in P:
                    if i in idx_sm_box_cl:
                        b = rois_obj_detec[i]
                        iou = RegionProposalExtraction.compute_iou(max_box_cl, b)

                        if iou >= self.config.NMS_THRESHOLD or RegionProposalExtraction.is_overlapping(max_box_cl, b):
                            P = np.delete(P, np.where(P == i))

        index_softmax_detec_final = index_softmax_detec[keep]
        rois_obj_detec_final = rois_obj_detec[keep]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(image)
        for i, (x_min, y_min, x_max, y_max) in enumerate(rois_obj_detec_final):
            w = x_max - x_min
            h = y_max - y_min

            index_class = index_softmax_detec_final[i]
            class_name = self.config.CLASSES[index_class]
            color = self.config.COLORS[index_class]

            ax.add_patch(plt.Rectangle((x_min, y_min), w, h, edgecolor=color, alpha=0.5, lw=2, facecolor='none'))
            ax.text(x_min, y_min, f"{class_name}", c=color, fontdict={'fontsize': 22})

        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    # rpe = RegionProposalExtraction('../../tasks/detection/data/images', '../../tasks/detection/data/annotations')
    # rpe.get_roi()
    # train_dataloader, val_dataloader = TrainDataLoader(batch_size=16, random_seed=42, valid_size=0.2,
    #                                                    shuffle=True).load_data()
    model = RCNN(pretrained=True)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.classifier.parameters(),
    #                              lr=model.config.LEARNING_RATE,
    #                              weight_decay=model.config.WEIGHT_DECAY)
    # model.train(train_dataloader, val_dataloader, criterion, optimizer, model.config.NUM_EPOCHS)
    img_path = '../../tasks/detection/fruits/data/images/fruit180.png'
    model.detect(img_path)
