import numpy as np
import os
import socket
import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from glob import glob
from PIL import Image
from .utils import get_data_folder,TransformTwice

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

class CIFAR100_ood(datasets.CIFAR100):
    def __init__(self, root, ood, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if ood == 'tin':
            data_path = '%s/%s/' % (root.replace('/cifar/', '/tinyImageNet200/'), 'train')
            self.img_list = [f for f in glob(data_path + "**/*.JPEG", recursive=True)]
        elif ood == 'places':
            data_path = root.replace('/cifar/', '/places365/')
            self.img_list = [f for f in glob(data_path + "**/*.jpg", recursive=True)]
        else:
            raise NotImplementedError

        self.img_list.sort()
        print('*********', ood, len(self.img_list))

        # mean predictions
        self.soft_labels = np.zeros((len(self.data) + len(self.img_list)), dtype=np.float32)
        for idx in range(len(self.data) + len(self.img_list)):
            if idx < len(self.data):
                self.soft_labels[idx] = 1.0
            else:
                self.soft_labels[idx] = 0

        # history predictions
        self.prediction = np.zeros((len(self.data) + len(self.img_list), 10), dtype=np.float32)
        self.prediction[:len(self.data), :] = 1.0
        self.count = 0

    def __len__(self):
        return len(self.data) + len(self.img_list)

    def label_update(self, results):
        self.count += 1
        # While updating the noisy label y_i by the probability s, we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 10
        self.prediction[len(self.data):, idx] = results[len(self.data):]
        if self.count >= 10:
            self.soft_labels = self.prediction.mean(axis=1)

    def __getitem__(self, index):
        if index < len(self.data):
            img = self.data[index]
            img = Image.fromarray(img)
        else:
            img = Image.open(self.img_list[index - len(self.data)])#!!!
            img = img.convert('RGB')
        return self.transform(img), self.soft_labels[index], index


class ood_loader(torch.utils.data.Dataset):
    def __init__(self, data_folder, ood, transform):
        self.transform = transform
        if ood == 'tin':
            data_path = '%s/tinyImageNet200/%s/' % (data_folder, 'train')
            self.img_list = [f for f in glob(data_path + "**/*.JPEG", recursive=True)]
        elif ood == 'places':
            data_path = '%s/%s/' % (data_folder, 'places365')
            self.img_list = [f for f in glob(data_path + "**/*.jpg", recursive=True)]
        self.img_list.sort()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        img = img.convert('RGB')
        return self.transform(img), index

def get_cifar100(transform_train=None, transform_val=None, out_dataset=None):
    data_folder = get_data_folder()
    assert out_dataset in ['tin', 'places']
    train_labeled_dataset = torchvision.datasets.CIFAR100(root=data_folder + '/cifar/', download=True, train=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=data_folder + '/cifar/', download=True, train=False, transform=transform_val)
    train_unlabeled_dataset = ood_loader(data_folder, ood=out_dataset, transform=TransformTwice(transform_train))
    train_dataset = CIFAR100_ood(root=data_folder + '/cifar/', ood=out_dataset, download=True, train=True,
                                 transform=transform_train)
    return train_labeled_dataset, train_unlabeled_dataset, train_dataset, test_dataset
