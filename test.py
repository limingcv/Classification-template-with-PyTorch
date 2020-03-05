"""
Author: liming
Email:  limingcv@qq.com
Github: https://github.com/limingcv
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms, models
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root_path="", transform=None, mode="test"):
        self.data_dir = os.path.join(root_path, mode)
        self.transform = transform
        self.raw_list = pd.read_csv(os.path.join(self.data_dir, "upload.csv"))
        self.images_list = [item[0] for item in np.array(self.raw_list)]
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_dir, self.images_list[index])).convert("L")

        if self.transform:
            image = self.transform(image)
        
        return image
    
    def __len__(self):
        return len(self.images_list)


def load_data_test(root_path, batch_size):
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.50501055,), std=(0.22912283,)),
    ])

    test_dataset = MyAIDataset(root_path, transforms_test)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return test_loader


def load_model(model_path, device):
    model = torch.load(model_path)
    model = model.to(device)
    return model


def write_results(root_path, model, test_loader, device, model_name):
    resluts = []
    for batch, img in enumerate(test_loader):
        img = img.to(device)
        batch_size = img.size(0)
        pred = model(img)
        _, pred = pred.topk(1)  # pred 是 (batch, 1) 的 tensor
        pred = pred.cpu().squeeze()
        for i in range(len(pred)):
            resluts.append(int(pred[i].item()))
        
    raw = pd.read_csv(os.path.join("upload.csv"))
    for i in range(len(resluts)):
        raw.iloc[i, 1] = int(resluts[i])
    raw.to_csv(model_name + ".csv", index=False)

    print("Done!")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_path = "/home/liming/code/python/dataset/flyai"
    model_path = "/home/liming/code/python/code/x-ray/logs/mobilenetv2_2020-03-01_19:58:00/mobilenetv2_12.pkl"
    model = load_model(model_path, device)
    test_loader = load_data_test(root_path, batch_size=16)

    write_results(root_path, model, test_loader, device, model_name="MobileNetv2"), 


main()