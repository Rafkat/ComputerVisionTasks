from collections import defaultdict

import pandas as pd
import torch
from torch import nn

from tasks.detection.fruits.training import SSDTraining, RCNNTraining
from tasks.detection.fruits.utils import MultiBoxLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tasks.utils import EarlyStopping


def ssd_training(n_epochs, device, lr):
    history = defaultdict(list)
    ssd_train = SSDTraining()
    ssd_loss = MultiBoxLoss(ssd_train.default_boxes)
    ssd_train.model.to(device)

    optimizer = torch.optim.Adam(ssd_train.model.parameters(), lr=lr)
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


def rcnn_training(n_epochs, device, lr):
    rcnn_train = RCNNTraining()
    history = rcnn_train.train(criterion=nn.CrossEntropyLoss(),
                               optimizer=torch.optim.Adam(rcnn_train.model.parameters(), lr=lr),
                               epochs=n_epochs)
    torch.save(rcnn_train.model.state_dict(), './rcnn_model.pth')
    pd.DataFrame(history).to_csv('./tasks/detection/fruits/logs/rcnn_history.csv', index=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 600
    lr = 1e-4

    ssd_training(n_epochs, device, lr)
    # rcnn_training(n_epochs, device, lr)


if __name__ == '__main__':
    main()
