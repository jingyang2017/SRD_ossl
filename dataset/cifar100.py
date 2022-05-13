from __future__ import print_function

import cv2
import numpy as np
import os
import socket
import torch
from PIL import Image
from glob import glob
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .utils import get_data_folder,TransformTwice

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)




# TIN: TinyImageNet
class TIN_loader(torch.utils.data.Dataset):
    def __init__(self, data_folder, transform):
        self.transform = transform
        data_path = '%s/tinyImageNet200/%s/' % (data_folder, 'train')
        self.img_list = [f for f in glob(data_path + "**/*.JPEG", recursive=True)]
        self.img_list.sort()

    def set_index(self, indexes=None):
        if indexes is not None:
            self.img_list = self.img_list[indexes]
        else:
            self.img_list = self.img_list

    def init_index(self):
        self.img_list = self.img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        img = img.convert('RGB')
        return self.transform(img), index



class Places365Instance(datasets.Places365):
    """places365Instance Dataset.
    """
    def __getitem__(self, index):
        file, target = self.imgs[index]
        image = self.loader(file)
        return self.transform(image), index

class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class InstanceSample(torch.utils.data.Dataset):
    """
    Instance+Sample Dataset
    """

    def __init__(self, root, data, model, transform, target_transform=None, k=4096, mode='exact', is_sample=True,
                 percent=1.0):

        if data == 'tin':
            infos = open('%s/tinyImageNet200/%s.txt' % (root, model)).readlines()
        else:
            raise NotImplementedError

        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []
        for info in infos:
            img_name, img_label = info.strip('\n').split(';')
            img = cv2.imread('/media/jd4615/dataB/Datasets/classification/' + img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.data.append(img)
            self.targets.append(int(img_label))

        cifar100_data = datasets.CIFAR100(root+'/cifar/', train=True, download=True)

        self.data.extend(list(cifar100_data.data))
        self.targets.extend(list(cifar100_data.targets))

        print(len(self.data),len(self.targets))

        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        num_samples = len(self.data)
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        num_samples = len(self.data)
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_cifar100_dataloaders(batch_size=128, num_workers=8,
                             is_instance=False,
                             is_sample=True,
                             k=4096, mode='exact', percent=1.0, ood='tin',model=None):
    """
    cifar 100
    """
    if is_instance: assert not is_sample
    if is_sample: assert not is_instance

    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    tiny_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4), # TODO
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    if is_instance and not is_sample:
        train_set = CIFAR100Instance(root=data_folder + '/cifar/',
                                     download=True,
                                     train=True,
                                     transform=train_transform)

    elif is_sample and not is_instance:
        train_set = InstanceSample(root=data_folder, data=ood, model=model, transform=tiny_transform, target_transform=None, k=k, mode=mode, is_sample=True,percent=percent)
    else:
        train_set = datasets.CIFAR100(root=data_folder + '/cifar/',
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True)

    test_set = datasets.CIFAR100(root=data_folder + '/cifar/',
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size / 2),
                             shuffle=False,
                             num_workers=int(num_workers / 2))

    if ood == 'tin':
        ood_set = TIN_loader(data_folder, transform=TransformTwice(tiny_transform))
    elif ood == 'places':
        ood_set = Places365Instance(data_folder + '/places365', transform=TransformTwice(tiny_transform))

    ood_loader = DataLoader(ood_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    if is_instance or is_sample:
        return train_loader, ood_loader, test_loader, n_data
    else:
        return train_loader, ood_loader, test_loader
