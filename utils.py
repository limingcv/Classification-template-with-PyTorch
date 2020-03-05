import numpy as np
from dataloader import *
from torchvision import transforms
import cv2
from PIL import Image
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from sklearn.metrics import roc_curve, auc


# root_path = "/home/liming/datasets/medium"
# mode = "train"
# train_dataset = MyDataset(root_path, transform=transforms.ToTensor())
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

def compute_mean_std(root_path, mode):
    pop_mean = []
    pop_std0 = []
    # print(dataset)
    for i, (img, label) in enumerate(train_loader):
        # print(img, label)
        # print(i, label)
        # shape (batch_size, 3, height, width)
        numpy_image = img.numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image)  # , axis=(0, 2, 3)
        batch_std0 = np.std(numpy_image)   #  , axis=(0, 2, 3)

        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    # print(pop_mean[0])
    # print(pop_std0[0])
    pop_mean = np.array(pop_mean).mean()
    pop_std0 = np.array(pop_std0).mean()

    print(pop_mean, pop_std0)


def compute_classes(train_loader):
    l = [0, 0, 0, 0]
    for i, (img, label) in enumerate(train_loader):
        l[label] += 1
    
    print(l)
    print(l / sum(np.array(l)))


def get_each_class_acc(labels, predictions):
    """compute each class's accuracy
    
    Arguments:
        labels {list} -- ground truth labels
        predictions {list} -- model output
    
    Returns:
        pred_classes {list} -- Number of correct predictions each class
        label_classes {list} -- Total number of each class
        ratio {list} -- Accuracy of each class
    """
    # type(labels): list, type(predictions): list
    num_classes = len(set(labels))

    pred_classes = [0 for _ in range(num_classes)] 
    label_classes = [0 for _ in range(num_classes)]
    ratio = [0 for _ in range(num_classes)] 
    
    for i in range(len(labels)):
        if (labels[i] == predictions[i]):
            pred_classes[labels[i]] += 1

        label_classes[labels[i]] += 1

    for i in range(num_classes):
        ratio[i] = pred_classes[i] / label_classes[i]
    
    return pred_classes, label_classes, ratio


def write_pr_curve(writer, labels, predictions, mode, epoch):
    num_classes = len(set(labels))

    l = [[] for _ in range(num_classes)]

    for i in range(len(labels)):
        l[labels[i]].append(int(predictions[i] == labels[i]))

    for i in range(num_classes):
        writer.add_pr_curve(mode + " dataset PR Curve for class " + str(i), np.array([1 for _ in range(len(l[i]))]), np.array(l[i]), epoch)
    
    
def write_acc_each_class(writer, ratio, epoch, mode):
    dicts = {}

    for i in range(len(ratio)):
        writer.add_scalar(mode + " dataset accuracy on class " + str(i), ratio[i], epoch-1)
    