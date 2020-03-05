import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, root_path="", transform=None, mode="train"):
        self.root_path = os.path.join(root_path, mode)
        self.transform = transform
        self.mode = mode
        self.raw_list = np.array(pd.read_csv(os.path.join(self.root_path, self.mode + ".csv")))
        self.images_list = [item[0] for item in self.raw_list]
        self.labels_list = [item[1] for item in self.raw_list]
    
    def __getitem__(self, index):
        images = Image.open(os.path.join(self.root_path, self.images_list[index])).convert("L")
        
        labels = self.labels_list[index]

        if self.transform:
            images = self.transform(images)

        return images, labels
    
    def __len__(self):
        return len(self.images_list)


def load_data(root_path, batch_size, img_size):
    train_transforms = transforms.Compose([
        # transforms.Resize((img_size, img_size)),
        # transforms.random
        transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.50501055,), std=(0.22912283,)),
        transforms.RandomErasing(),
    ])

    val_transforms = transforms.Compose([
        # transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.50501055,), std=(0.22912283,)),
    ])


    train_dataset = MyDataset(root_path, train_transforms, "train")
    val_dataset = MyDataset(root_path, val_transforms, "val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
