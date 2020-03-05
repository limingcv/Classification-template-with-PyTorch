"""
Author: liming
Email:  limingcv@qq.com
Github: https://github.com/limingcv
"""

import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import logging
import argparse
from dataloader import load_data
from torchvision import models
from loghelp import Logger
import time
from focal_loss import FocalLoss
from models.cbam_resnext import * 
from models.se_resnext import *
from models.se_resnet import *
from torch.utils.tensorboard import SummaryWriter
import utils


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--epochs', '-e', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', '-b', default=128, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--weight_decay', '-wd', default=0.05, type=float, help='weight decay in optimizer')
parser.add_argument('--model_name', default="resnet18", type=str, help='model name')
parser.add_argument('--root_path', default="/home/liming/datasets/x-ray", type=str, help='root path to your dataset')
parser.add_argument('--save_path', default="saved_models", type=str, help='directory to save model')
parser.add_argument('--description', default="focal loss, data augument RandomErasing, ColorJitter(1), Adam, lr decay", type=str, help='description')
parser.add_argument('--lr_decay_step', default="10", type=int, help='how many epochs to decay the learning rate')
parser.add_argument('--lr_decay_rate', default="0.1", type=float, help='multiple of learning rate decay')
parser.add_argument('--img_size', default=112, type=int, help='image size you want to resize')



def train(train_loader, model, critetion, optimizer, epoch):
    model.train()

    train_acc_sum, n =  0, 0
    running_loss = 0.0

    predictions, labels = [], []

    for batch, data in enumerate(train_loader):
        img, label = data[0], data[1]
        img, label = img.to(device), label.to(device)

        output = model(img)
        _, pred = output.topk(1)
        pred = pred.t()
        n += img.shape[0]
        train_acc_sum += (pred == label).float().sum().item()

        loss = critetion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.flatten(pred)
        for pred in preds:
            predictions.append(pred.item())

        label = torch.flatten(label)
        for x in label:
            labels.append(x.item())

        running_loss += loss.item()

        if batch % 50 == 0:
            line = "Epoch: [{}/{}]\t Batch[{}/{}]\t Loss: {:.5f}\t Train Accuracy: {:.5f}".format( \
                    epoch, args.epochs, batch, len(train_loader), loss.item(), train_acc_sum / n)
            logger.Print(line)

    _, _, ratio = utils.get_each_class_acc(labels, predictions)
    utils.write_acc_each_class(writer, ratio, epoch, mode="train")

    line1 = "Epoch: [{}/{}]\t Train Accuracy: {}\t Train Loss: {}".format(epoch, args.epochs, train_acc_sum / n, running_loss / n)
    line2 = "Train accuracy on different classes: {}".format(ratio)
    logger.Print(line1)
    logger.Print(line2)

    return train_acc_sum / n, running_loss / n


def validate(val_loader, model, critetion, epoch):
    model.eval()
    running_loss = 0.0
    predictions, labels = [], []

    with torch.no_grad():
        val_acc_sum, n =  0, 0
        for batch, data in enumerate(val_loader):
            img, label = data[0], data[1]
            img, label = img.to(device), label.to(device)
            output = model(img)
            _, pred = output.topk(1)
            pred = pred.t()
            loss = critetion(output, label)
                
            running_loss += loss.item()

            val_acc_sum += (pred == label).float().sum().item()
            n += img.shape[0]
    
            preds = torch.flatten(pred)
            for pred in preds:
                predictions.append(pred.item())

            label = torch.flatten(label)
            for x in label:
                labels.append(x.item())

    line1 = "Epoch: [{}/{}]\t Validation Accuracy: {}\t Validation Loss: {}".format(epoch, args.epochs, val_acc_sum / n, running_loss / n)
    _, _, ratio = utils.get_each_class_acc(labels, predictions)
    utils.write_acc_each_class(writer, ratio, epoch, mode="val")
    line2 = "Val accuracy on different classes: {}".format(ratio)
    logger.Print(line1)
    logger.Print(line2)
    logger.Print("======================================================================================================================")

    return val_acc_sum / n, running_loss / n


def main():
    global args, device, root_path, writer

    train_loader, val_loader = load_data(root_path, batch_size=args.batch_size, img_size=args.img_size)

    model = models.resnet18(pretrained=True)
    conv1 = nn.Conv2d(1, 3, kernel_size = 1, bias=False)
    model.fc = nn.Linear(in_features=512, out_features=4, bias=False)
    model = nn.Sequential(conv1, model)

    model = model.to(device)

    # nums of four types in train dataset: [4795, 2569, 2395, 4743]
    # rate of four types in train dataset: [0.33064404909667633, 0.1771479795890222, 0.16514963453316783, 0.32705833678113366]
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(gamma=0, alpha=[0.23889842593449953, 0.3152914838897476, 0.21944465598406257, 0.22636543419169028])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_acc = 0.0
    line = "Start training {} on {}...".format(args.model_name, device)
    logger.Print(line)

    # print training parameters
    logger.Print("======================================================================================================================")
    logger.Print("lr: {}".format(args.lr))
    logger.Print("batch size: {}".format(args.batch_size))
    logger.Print("weight decay: {}".format(args.weight_decay))
    logger.Print("model: {}".format(args.model_name))
    logger.Print("epochs: {}".format(args.epochs))
    logger.Print("description: {}".format(args.description))
    logger.Print("lr decay interval: {}".format(args.lr_decay_step))
    logger.Print("lr decay: {}".format(args.lr_decay_rate))
    logger.Print("======================================================================================================================")

    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, gamma=args.lr_decay_rate)
    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs+1):
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
        val_acc, val_loss = validate(val_loader, model, criterion, epoch)
        scheduler.step()  # in PyTorch 1.1 and later, `optimizer.step()` should be called before `lr_scheduler.step()`

        # write loss and accuracy to tensorboard
        writer.add_scalars(args.model_name + ' Accuracy', {'train accuracy': train_acc, 'val accuracy': val_acc}, epoch-1)
        writer.add_scalars(args.model_name + ' Loss', {'train loss': train_loss, 'val loss': val_loss}, epoch-1)

        # record the best result on validation dataset
        if val_acc > best_acc:
            logger.Print('epoch: {} The best is {} last best is {}'.format(epoch, val_acc, best_acc))
            best_acc = val_acc
            modelname = "{}/{}_{}.pkl".format(log_path, args.model_name, epoch)
            torch.save(model, modelname)
            line = "Have saved model with acc {} to {}".format(best_acc, log_path)
            logger.Print(line)
            logger.Print("======================================================================================================================")

    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_path=args.root_path
    writer = SummaryWriter()

    time_stp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    
    log_path = f"logs/{args.model_name}_{time_stp}/"
    if not os.path.exists(log_path):
        os.makedirs(log_path,mode=0o777)
    logger = Logger(log_path + '/log.log')
    
    main()

# Open tensorboard in a browser using the command line: 
# tensorboard --logdir runs
