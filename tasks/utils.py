import torch
from torch import nn

from models.detection_convolution.SSD_utils import cxcy_to_xy, find_IoU, encode_bboxes, xy_to_cxcy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def combine(batch):
    images = []
    boxes = []
    labels = []

    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])

    images = torch.stack(images, dim=0)
    return images, boxes, labels


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


class MultiBoxLoss(nn.Module):
    def __init__(self, default_boxes, threshold=0.5, neg_pos=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.default_boxes = default_boxes
        self.threshold = threshold
        self.neg_pos = neg_pos
        self.alpha = alpha

    def forward(self, locs_pred, cls_pred, boxes, labels):
        batch_size = locs_pred.size(0)
        n_default_boxes = self.default_boxes.size(0)
        num_classes = cls_pred.size(2)

        target_locs = torch.zeros((batch_size, n_default_boxes, 4), dtype=torch.float).to(device)
        target_classes = torch.zeros((batch_size, n_default_boxes), dtype=torch.long).to(device)

        default_boxes_xy = cxcy_to_xy(self.default_boxes)
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_IoU(boxes[i], default_boxes_xy)

            overlap_each_default_box, object_each_default_box = overlap.max(dim=0)

            _, default_boxes_each_object = overlap.max(dim=1)

            object_each_default_box[default_boxes_each_object] = torch.LongTensor(range(n_objects)).to(device)

            overlap_each_default_box[default_boxes_each_object] = 1.

            label_each_default_box = labels[i][object_each_default_box]

            label_each_default_box[overlap_each_default_box < self.threshold] = 0

            target_classes[i] = label_each_default_box

            target_locs[i] = encode_bboxes(xy_to_cxcy(boxes[i][object_each_default_box]), self.default_boxes)

        pos_default_boxes = target_classes != 0

        smooth_L1_loss = nn.SmoothL1Loss()
        loc_loss = smooth_L1_loss(locs_pred[pos_default_boxes], target_locs[pos_default_boxes])

        n_positive = pos_default_boxes.sum(dim=1)
        n_hard_negatives = self.neg_pos * n_positive

        cross_entropy_loss = nn.CrossEntropyLoss(reduce=False)
        confidence_loss_all = cross_entropy_loss(cls_pred.view(-1, num_classes), target_classes.view(-1))
        confidence_loss_all = confidence_loss_all.view(batch_size, n_default_boxes)

        confidence_pos_loss = confidence_loss_all[pos_default_boxes]

        confidence_neg_loss = confidence_loss_all.clone()
        confidence_neg_loss[pos_default_boxes] = 0
        confidence_neg_loss, _ = confidence_neg_loss.sort(dim=1, descending=True)

        hardness_ranks = torch.LongTensor(range(n_default_boxes)).unsqueeze(0).expand_as(confidence_neg_loss).to(
            device)

        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)

        confidence_hard_neg_loss = confidence_neg_loss[hard_negatives]

        confidence_loss = (confidence_hard_neg_loss.sum() + confidence_pos_loss.sum()) / n_positive.sum().float()

        return self.alpha * loc_loss, confidence_loss


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties,
                  label_map, rev_label_map):
    assert (len(det_boxes) == len(det_labels) == len(det_scores)
            == len(true_boxes) == len(true_labels) == len(true_difficulties))
    # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = []
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = []
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_IoU(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0))).to(device)[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[
                            original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision
