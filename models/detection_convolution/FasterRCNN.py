import os

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from bs4 import BeautifulSoup
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import ops
from torchvision import transforms

from models.detection_convolution.FasterRCNN_utils import generate_proposals, gen_anc_centers, gen_anc_base, \
    project_bboxes, get_req_anchors, calc_cls_loss, calc_bbox_reg_loss


# originated from https://arxiv.org/pdf/1506.01497


class TrainDataset(Dataset):
    def __init__(self, classes_map, image_dir='./tasks/detection/fruits/data/images',
                 annot_dir='./tasks/detection/fruits/data/annotations', transform=None, image_size=(480, 640)):
        self.transform = transform
        self.img_height, self.img_width = image_size

        self.image_dir = image_dir
        self.annot_dir = annot_dir

        self.image_paths = list(sorted(os.listdir(image_dir)))
        self.annot_paths = list(sorted(os.listdir(annot_dir)))
        self.classes_map = classes_map

        self.gt_bboxes_all, self.gt_classes_all, self.gt_difficulties_all = self.get_data()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.gt_bboxes_all[idx], self.gt_classes_all[idx], self.gt_difficulties_all[idx]

    def get_data(self):
        gt_boxes_all = []
        gt_classes_all = []
        gt_difficulties_all = []

        for idx in range(len(self.annot_paths)):
            annot_path = self.annot_paths[idx]
            annot = open(os.path.join(self.annot_dir, annot_path)).read()
            soup = BeautifulSoup(annot, 'html.parser')
            classes = []
            bboxes = []
            difficulties = []
            img_width, img_height = int(soup.find('width').string), int(soup.find('height').string)

            for obj in soup.find_all('object'):
                classes.append(self.classes_map[obj.find('name').string])
                difficulties.append(int(obj.find('difficult').string))

                x_min = max(0, int(obj.find('xmin').string))
                y_min = max(0, int(obj.find('ymin').string))
                x_max = min(img_width, int(obj.find('xmax').string))
                y_max = min(img_height, int(obj.find('ymax').string))

                bbox = [x_min, y_min, x_max, y_max]
                if self.transform:
                    x_min = x_min * self.img_width / img_width
                    x_max = x_max * self.img_width / img_width
                    y_min = y_min * self.img_height / img_height
                    y_max = y_max * self.img_height / img_height
                    bbox = [x_min, y_min, x_max, y_max]
                bboxes.append(bbox)

            gt_boxes_all.append(torch.Tensor(bboxes))
            gt_classes_all.append(torch.Tensor(classes))
            gt_difficulties_all.append(torch.Tensor(difficulties))

        gt_bboxes_pad = pad_sequence(gt_boxes_all, batch_first=True, padding_value=-1)
        gt_classes_pad = pad_sequence(gt_classes_all, batch_first=True, padding_value=-1)
        gt_difficulties_pad = pad_sequence(gt_difficulties_all, batch_first=True, padding_value=-1)
        return gt_bboxes_pad, gt_classes_pad, gt_difficulties_pad


class FasterRCNNTrainDataLoader:
    def __init__(self, classes_map, batch_size=128, random_seed=42, valid_size=0.2, shuffle=True):
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.valid_size = valid_size
        self.shuffle = shuffle
        self.classes_map = classes_map

    def load_data(self, image_dir, annot_dir):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        valid_transform = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_dataset = TrainDataset(image_dir=image_dir, annot_dir=annot_dir,
                                     transform=train_transform, classes_map=self.classes_map)
        valid_dataset = TrainDataset(image_dir=image_dir, annot_dir=annot_dir,
                                     transform=valid_transform, classes_map=self.classes_map)

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
                                  num_workers=4, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, sampler=valid_sample,
                                  num_workers=4, pin_memory=True)
        return train_loader, valid_loader


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        req_layers = list(model.children())[:8]
        self.backbone = nn.Sequential(*req_layers)
        for param in self.backbone.named_parameters():
            param[1].requires_grad = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone.to(self.device)

    def forward(self, img_data):
        return self.backbone(img_data)


class ProposalModule(nn.Module):
    def __init__(self, in_features, hidden_dim=512, n_anchors=9, p_dropout=0.3):
        super().__init__()
        self.n_anchors = n_anchors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=3, padding=1).to(self.device)
        self.dropout = nn.Dropout(p_dropout).to(self.device)
        self.conf_head = nn.Conv2d(hidden_dim, n_anchors, kernel_size=1).to(self.device)
        self.reg_head = nn.Conv2d(hidden_dim, n_anchors * 4, kernel_size=1).to(self.device)

    def forward(self, feature_map, pos_anc_ind=None, neg_anc_ind=None, pos_anc_coords=None):
        # determine mode
        if pos_anc_ind is None or neg_anc_ind is None or pos_anc_coords is None:
            mode = 'eval'
        else:
            mode = 'train'

        out = self.conv1(feature_map)
        out = F.relu(self.dropout(out))

        reg_offsets_pred = self.reg_head(out).cpu()  # (B, A*4, hmap, wmap)
        conf_scores_pred = self.conf_head(out).cpu()  # (B, A, hmap, wmap)

        if mode == 'train':
            # get conf scores
            conf_scores_pos = conf_scores_pred.flatten()[pos_anc_ind]
            conf_scores_neg = conf_scores_pred.flatten()[neg_anc_ind]
            # get offsets for +ve anchors
            offsets_pos = reg_offsets_pred.contiguous().view(-1, 4)[pos_anc_ind]
            # generate proposals using offsets
            proposals = generate_proposals(pos_anc_coords, offsets_pos)

            return conf_scores_pos, conf_scores_neg, offsets_pos, proposals

        elif mode == 'eval':
            return conf_scores_pred, reg_offsets_pred


class RegionProposalNetwork(nn.Module):
    def __init__(self, img_size, out_size, out_channels):
        super().__init__()

        self.img_height, self.img_width = img_size
        self.out_h, self.out_w = out_size

        # downsampling scale factor
        self.width_scale_factor = self.img_width // self.out_w
        self.height_scale_factor = self.img_height // self.out_h

        # scales and ratios for anchor boxes
        self.anc_scales = [2, 4, 6]
        self.anc_ratios = [0.5, 1, 1.5]
        self.n_anc_boxes = len(self.anc_scales) * len(self.anc_ratios)

        # IoU thresholds for +ve and -ve anchors
        self.pos_thresh = 0.7
        self.neg_thresh = 0.3

        # weights for loss
        self.w_conf = 1
        self.w_reg = 5

        self.feature_extractor = FeatureExtractor()
        self.proposal_module = ProposalModule(out_channels, n_anchors=self.n_anc_boxes)

    def forward(self, images, gt_bboxes, gt_classes):
        batch_size = images.size(dim=0)
        feature_map = self.feature_extractor(images)

        # generate anchors
        anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(self.out_h, self.out_w))
        anc_base = gen_anc_base(anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
        anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)

        # get positive and negative anchors amongst other things
        gt_bboxes_proj = project_bboxes(gt_bboxes, self.width_scale_factor, self.height_scale_factor, mode='p2a')

        positive_anc_ind, negative_anc_ind, _, GT_offsets, GT_class_pos, positive_anc_coords, \
            _, positive_anc_ind_sep = get_req_anchors(anc_boxes_all, gt_bboxes_proj, gt_classes)

        # pass through the proposal module
        conf_scores_pos, conf_scores_neg, offsets_pos, proposals = self.proposal_module(feature_map, positive_anc_ind,
                                                                                        negative_anc_ind,
                                                                                        positive_anc_coords)

        cls_loss = calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size)
        reg_loss = calc_bbox_reg_loss(GT_offsets, offsets_pos, batch_size)

        total_rpn_loss = self.w_conf * cls_loss + self.w_reg * reg_loss

        return total_rpn_loss, feature_map, proposals, positive_anc_ind_sep, GT_class_pos

    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        with torch.no_grad():
            batch_size = images.size(dim=0)
            feature_map = self.feature_extractor(images)

            # generate anchors
            anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(self.out_h, self.out_w))
            anc_base = gen_anc_base(anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
            anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)
            anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)

            # get conf scores and offsets
            conf_scores_pred, offsets_pred = self.proposal_module(feature_map)
            conf_scores_pred = conf_scores_pred.reshape(batch_size, -1)
            offsets_pred = offsets_pred.reshape(batch_size, -1, 4)

            # filter out proposals based on conf threshold and nms threshold for each image
            proposals_final = []
            conf_scores_final = []
            for i in range(batch_size):
                conf_scores = torch.sigmoid(conf_scores_pred[i])
                offsets = offsets_pred[i]
                anc_boxes = anc_boxes_flat[i]
                proposals = generate_proposals(anc_boxes, offsets)
                # filter based on confidence threshold
                conf_idx = torch.where(conf_scores >= conf_thresh)[0]
                conf_scores_pos = conf_scores[conf_idx]
                proposals_pos = proposals[conf_idx]
                # filter based on nms threshold
                nms_idx = ops.nms(proposals_pos, conf_scores_pos, nms_thresh)
                conf_scores_pos = conf_scores_pos[nms_idx]
                proposals_pos = proposals_pos[nms_idx]

                proposals_final.append(proposals_pos)
                conf_scores_final.append(conf_scores_pos)

        return proposals_final, conf_scores_final, feature_map


class ClassificationModule(nn.Module):
    def __init__(self, out_channels, n_classes, roi_size, hidden_dim=512, p_dropout=0.3):
        super().__init__()
        self.roi_size = roi_size
        # hidden network
        self.avg_pool = nn.AvgPool2d(self.roi_size)
        self.fc = nn.Linear(out_channels, hidden_dim)
        self.dropout = nn.Dropout(p_dropout)

        # define classification head
        self.cls_head = nn.Linear(hidden_dim, n_classes)

    def forward(self, feature_map, proposals_list, gt_classes=None):

        if gt_classes is None:
            mode = 'eval'
        else:
            mode = 'train'

        # apply roi pooling on proposals followed by avg pooling
        roi_out = ops.roi_pool(feature_map, proposals_list, self.roi_size)
        roi_out = self.avg_pool(roi_out)

        # flatten the output
        roi_out = roi_out.squeeze(-1).squeeze(-1)

        # pass the output through the hidden network
        out = self.fc(roi_out)
        out = F.relu(self.dropout(out))

        # get the classification scores
        cls_scores = self.cls_head(out)

        if mode == 'eval':
            return cls_scores

        # compute cross entropy loss
        cls_loss = F.cross_entropy(cls_scores, gt_classes.long())

        return cls_loss


class FasterRCNN(nn.Module):
    def __init__(self, img_size=(480, 640), out_size=(15, 20), out_channels=2048, n_classes=5, roi_size=(2, 2)):
        super().__init__()
        self.rpn = RegionProposalNetwork(img_size, out_size, out_channels)
        self.classifier = ClassificationModule(out_channels, n_classes, roi_size)

    def forward(self, images, gt_bboxes, gt_classes):
        total_rpn_loss, feature_map, proposals, positive_anc_ind_sep, GT_class_pos = self.rpn(images, gt_bboxes,
                                                                                              gt_classes)

        # get separate proposals for each sample
        pos_proposals_list = []
        batch_size = images.size(dim=0)
        for idx in range(batch_size):
            proposal_idxs = torch.where(positive_anc_ind_sep == idx)[0]
            proposals_sep = proposals[proposal_idxs].detach().clone()
            pos_proposals_list.append(proposals_sep)

        cls_loss = self.classifier(feature_map.cpu(), pos_proposals_list, GT_class_pos)
        total_loss = cls_loss + total_rpn_loss

        return total_loss

    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        batch_size = images.size(dim=0)
        proposals_final, conf_scores_final, feature_map = self.rpn.inference(images, conf_thresh, nms_thresh)
        cls_scores = self.classifier(feature_map.cpu(), proposals_final)

        # convert scores into probability
        cls_probs = F.softmax(cls_scores, dim=-1)
        # get classes with the highest probability
        classes_all = torch.argmax(cls_probs, dim=-1)

        classes_final = []
        # slice classes to map to their corresponding image
        c = 0
        for i in range(batch_size):
            n_proposals = len(proposals_final[i])  # get the number of proposals for each image
            classes_final.append(classes_all[c: c + n_proposals])
            c += n_proposals

        return proposals_final, conf_scores_final, classes_final


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FasterRCNN()
    model.eval()
    # model.load_state_dict(torch.load("../../tasks/detection/fruits/faster_rcnn.pth"))
    model.load_state_dict(torch.load("../../faster_rcnn.pth"))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    from torchvision import transforms

    image = Image.open('../../tasks/detection/fruits/data/images/fruit110.png').convert('RGB')
    resize = transforms.Resize((480, 640))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=mean, std=std)
    classes_map = {'background': 0, 'pineapple': 1, 'snake fruit': 2, 'dragon fruit': 3, 'banana': 4}
    rev_class_map = {v: k for k, v in classes_map.items()}

    img = resize(image)
    img = to_tensor(img)
    img = normalize(img)
    proposals, conf_scores, classes = model.inference(img.unsqueeze(0).to(device), conf_thresh=0.9, nms_thresh=0.05)

    proposals = proposals[0]
    conf_scores = conf_scores[0]
    classes = classes[0]

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('../../tasks/detection/PascalVOC/arial.ttf', 15)
    for i in range(len(proposals)):
        box_location = proposals[i].tolist()
        box_location[0] *= image.width / 20
        box_location[1] *= image.height / 15
        box_location[2] *= image.width / 20
        box_location[3] *= image.height / 15
        draw.rectangle(xy=box_location, outline='red', width=2)

        text_size = font.getbbox(rev_class_map[classes[i].item()].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1],
                            box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, outline='red', width=2)
        draw.text(xy=text_location, text=rev_class_map[classes[i].item()].upper(), fill='white', font=font)

    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()
