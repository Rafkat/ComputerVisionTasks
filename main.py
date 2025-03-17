import torch
from torch import nn

from convolution_models.ResNet50 import ResNet50
from trainings.cifar10.training import Cifar10Training

dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet50 = ResNet50(nb_classes=10).to(dvc)
training = Cifar10Training(resnet50)
train_dataloader, val_dataloader = training.load_cifar_data(img_size=224,
                                                            batch_size=32)
optimizer = torch.optim.Adam(resnet50.parameters(), lr=1e-4)
training.train(train_dataloader, val_dataloader, optimizer,
               loss_func=nn.CrossEntropyLoss(), epochs=50, device=dvc, postfix='batch_reduced')


