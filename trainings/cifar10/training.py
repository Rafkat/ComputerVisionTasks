import os

import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn
from tqdm import tqdm

from trainings.cifar10.model_configs import models_config


class Cifar10Training:
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

    def load_cifar_data(self, img_size=32, batch_size=32):
        mean = [0.4670, 0.4735, 0.4662]
        std = [0.2496, 0.2489, 0.2551]

        transformations = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transformations, download=True)
        val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transformations, download=True)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True if self.device == 'cuda' else False
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True if self.device == 'cuda' else False
        )

        return train_dataloader, val_dataloader

    def train(self, train_dataloader, val_dataloader, optimizer, loss_func, epochs, device, postfix='v1'):
        print('Start training...')
        history = pd.DataFrame(columns=['loss', 'acc', 'val_loss', 'val_acc'])
        print('Training')
        for epoch in range(epochs):
            self.model.train()

            correct_train = 0
            total_train = 0
            for i, (images, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                images = images.to(device)
                labels = labels.to(device)

                predictions = self.model(images)
                loss = loss_func(predictions, labels)
                total_train += labels.size(0)
                correct_train += (torch.argmax(predictions, dim=1) == labels).sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_acc = correct_train / total_train * 100

            with torch.no_grad():
                self.model.eval()
                correct = 0
                total = 0
                all_val_loss = []

                for images, labels in val_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    predictions = self.model(images)

                    total += labels.size(0)
                    predicted = torch.argmax(predictions, dim=1)
                    correct += (predicted == labels).sum().item()
                    all_val_loss.append(loss_func(predictions, labels).item())

                mean_val_loss = sum(all_val_loss) / len(all_val_loss)
                mean_val_acc = correct / total * 100

            history.loc[epoch + 1] = [loss.item(), train_acc, mean_val_loss, mean_val_acc]

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}, "
                  f"Val-loss: {mean_val_loss}, Val-acc: {mean_val_acc}")

        history.to_csv(f'trainings/cifar10/logs/{self.model.__class__.__name__}_history_{postfix}.csv',
                       index=False)
        print(f'Finished Training {self.model.__class__.__name__}')
        torch.save(self.model.state_dict(), f'trainings/cifar10/weights/{self.model.__class__.__name__}.pth')
        print(f'Model weights saved to {self.model.__class__.__name__}.pth')


if __name__ == '__main__':
    dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model_config in models_config.values():
        model = model_config['model']
        training = Cifar10Training(model.to(dvc))
        train_dataloader, val_dataloader = training.load_cifar_data(img_size=model_config['img_size'],
                                                                    batch_size=model_config['batch_size'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        training.train(train_dataloader, val_dataloader, optimizer,
                       loss_func=nn.CrossEntropyLoss(), epochs=50, device=dvc)
