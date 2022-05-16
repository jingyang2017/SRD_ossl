from __future__ import print_function
import os
import argparse
import socket
import time
import torch
import torch.nn as nn
import numpy as np
from glob import glob
from torchvision import transforms
from models import model_dict
from PIL import Image

torch.backends.cudnn.benchmark = True

class TIN_loader(torch.utils.data.Dataset):
    def __init__(self, data_folder, transform):
        self.transform = transform
        data_path = '%s/tinyImageNet200/%s/' % (data_folder, 'train')
        self.img_list = [f for f in glob(data_path + "**/*.JPEG", recursive=True)]
        self.img_list.sort()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        img = img.convert('RGB')
        return self.transform(img), self.img_list[index]


parser = argparse.ArgumentParser(description='test model on ood data')
parser.add_argument('--model', type=str, choices=['resnet32x4', 'wrn_40_2'])
args = parser.parse_args()
# load model
if args.model == 'wrn_40_2':
    pretrain = './save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth'
elif args.model == 'resnet32x4':
    pretrain = './save/models/resnet32x4_vanilla/ckpt_epoch_240.pth'
else:
    raise NotImplementedError

model = model_dict[args.model](num_classes=100)
model.load_state_dict(torch.load(pretrain, 'cpu')['model'])
model = model.cuda()
model.eval()
# load data
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

tiny_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std),
])

tin_data = TIN_loader(data_folder='/media/jd4615/dataB/Datasets/classification/', transform=tiny_transform)
print(len(tin_data))
tin_loader = torch.utils.data.DataLoader(tin_data, batch_size=64, shuffle=False, num_workers=4, drop_last=False)
# assign label
f = open('/media/jd4615/dataB/Datasets/classification/tinyImageNet200/%s.txt'%args.model,'w')
for index, (imgs,img_names) in enumerate(tin_loader):
    imgs = imgs.cuda()
    with torch.no_grad():
        outputs = model(imgs)
    preds = torch.max(outputs, 1)[1]
    preds = np.array(preds.cpu().data)
    preds = list(preds)
    for img_name, label in zip(img_names, preds):
        f.write(img_name.replace('/media/jd4615/dataB/Datasets/classification/',''))
        f.write(';')
        f.write(str(label))
        f.write('\n')
f.close()

