from collections import defaultdict

import pandas as pd
import torch
from torch import nn

from models.detection_convolution.SSD import SingleShotMultiBoxDetector
from tasks.detection.PascalVOC.training import PascalVOCTraining
from tasks.detection.fruits.training import SSDTraining, RCNNTraining
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tasks.utils import EarlyStopping, MultiBoxLoss


def ssd_training(n_epochs, device, lr, ds='fruits'):
    if ds == 'fruits':  # Not trainable, too few examples in dataset fruits
        history = defaultdict(list)
        ssd_train = SSDTraining()
        ssd_loss = MultiBoxLoss(ssd_train.default_boxes)
        ssd_train.model.to(device)

        optimizer = torch.optim.Adam(ssd_train.model.parameters(), lr=lr, weight_decay=0.1)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
        early_stopping = EarlyStopping(patience=10)
        for epoch in range(n_epochs):
            total_loss, APs, mAP = ssd_train.train(ssd_loss, optimizer=optimizer)
            scheduler.step(total_loss)

            history['loss'].append(total_loss)
            history['pineapple_AP'].append(APs['pineapple'])
            history['snake_fruit_AP'].append(APs['snake fruit'])
            history['dragon_fruit_AP'].append(APs['dragon fruit'])
            history['banana_AP'].append(APs['banana'])
            history['mAP'].append(mAP)

            print(f'[INFO]: Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss:.4f}, APs: {APs}, mAP: {mAP}')
            early_stopping(total_loss, ssd_train.model)
            if early_stopping.early_stop:
                print('Early stopping')
                break

        pd.DataFrame(history).to_csv('./tasks/detection/fruits/logs/ssd_history.csv', index=False)
        torch.save(ssd_train.model.state_dict(), './ssd_model.pth')
    elif ds == 'voc':
        ssd_model = SingleShotMultiBoxDetector(nb_classes=21)
        training = PascalVOCTraining(ssd_model)
        training.train(n_epochs, lr=lr, wd=0, postfix='v0')


def rcnn_training(n_epochs, device, lr):
    rcnn_train = RCNNTraining()
    history = rcnn_train.train(criterion=nn.CrossEntropyLoss(),
                               optimizer=torch.optim.Adam(rcnn_train.model.parameters(), lr=lr),
                               epochs=n_epochs)
    torch.save(rcnn_train.model.state_dict(), 'models/detection_convolution/rcnn_model.pth')
    pd.DataFrame(history).to_csv('./tasks/detection/fruits/logs/rcnn_history.csv', index=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 600
    lr = 1e-3

    ssd_training(n_epochs, device, lr, ds='voc')
    # rcnn_training(n_epochs, device, lr)


if __name__ == '__main__':
    main()
