import os
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms  as transforms
from torch.utils.data import dataloader as DataLoader
from resnext import *
from resnet import *
from seresnext import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='/home/liming/datasets/cifar10/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])), batch_size=4, shuffle=True, num_workers=4)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='/home/liming/datasets/cifar10/', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ])), batch_size=4, shuffle=True, num_workers=4)





def train(epoch, model, name):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Model: {}  Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                name, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, name):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nModel: {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(name, test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))


models = {'resnet50': resnet50(),
          'resnext50': resnext50(),
          'seresnet50': se_resnext50(),
          'seresnext50': se_resnext50(),
          'skresnet50': skresnet50(),
          'skresnext50': skresnext50(),
          'cbam_resnext50': cbam_resnext50(),
          'cbam_resnext50': cbam_resnext50(),}

for model_name in models:
    optimizer = torch.optim.SGD(models[model_name].parameters(), lr=0.1, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 100

    for epoch in range(num_epochs):
        train(epoch, models[model_name].to(device), name=model_name)
        test(models[model_name].to(device), name=model_name)

