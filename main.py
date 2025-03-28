import torch
from torch import nn

from tasks.classification.cifar10.model_configs import models_config
from tasks.classification.cifar10.training import Cifar10Training

dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for model_config in models_config.values():
    model = model_config['model']
    training = Cifar10Training(model.to(dvc))
    train_dataloader, val_dataloader = training.load_cifar_data(img_size=model_config['img_size'],
                                                                batch_size=model_config['batch_size'])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    training.train(train_dataloader, val_dataloader, optimizer,
                   loss_func=nn.CrossEntropyLoss(), epochs=50, device=dvc, postfix='early_stop_sgd')
