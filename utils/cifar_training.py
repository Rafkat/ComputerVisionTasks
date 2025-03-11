import os

import torch
from datasets import load_dataset
import torchvision.transforms as transforms


class Cifar10Training:
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

    def preprocess_images(self, images, transformations):
        inputs = []
        for record in images:
            image = record['img']
            label = record['label']

            if image.mode == 'L':
                image = image.convert('RGB')

            input_tensor = transformations(image)
            inputs.append([input_tensor, label])

        return inputs

    def load_cifar_data(self, img_size=32, batch_size=32):
        dataset = load_dataset('cifar10')
        train_dataset, val_dataset = dataset['train'], dataset['val']

        mean = [0.4670, 0.4735, 0.4662]
        std = [0.2496, 0.2489, 0.2551]

        transformations = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        inputs_train = self.preprocess_images(train_dataset, transformations)
        inputs_val = self.preprocess_images(val_dataset, transformations)

        train_dataloader = torch.utils.data.DataLoader(
            inputs_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True if self.device == 'cuda' else False
        )

        val_dataloader = torch.utils.data.DataLoader(
            inputs_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True if self.device == 'cuda' else False
        )

        return train_dataloader, val_dataloader

    def train(self, train_dataloader, val_dataloader, optimizer, loss_func, epochs, device):
        for epoch in range(epochs):
            self.model.train()

            for i, (images, labels) in enumerate(train_dataloader):
                images = images.to(device)
                labels = labels.to(device)

                predictions = self.model(images)
                loss = loss_func(predictions, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
                mean_val_acc = 100 * (correct / total)

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}, "
                  f"Val-loss: {mean_val_loss}, Val-acc: {mean_val_acc}")


