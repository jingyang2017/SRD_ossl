import os
import socket

import torchvision.datasets as datasets


class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        return self.transform(img),self.target_transform(img),index

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('jd4615'):
        data_folder = '/media/jd4615/dataB/Datasets/classification/'
    else:
        #TODO put your dataset here
        data_folder = '/fsx/jinang/Datasets/classification/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder
