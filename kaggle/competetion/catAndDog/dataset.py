# -*- coding:utf-8 -*-

import os
import re

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# r'E:\DataSet\DataSet\kaggle\get_started\CatAndDog\train'
class CatDog(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = os.listdir(root)
        self.images.sort(lambda x: int(re.findall(r'\d+', x)[0]))  #

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file = self.images[index]
        image = np.array(Image.open(os.path.join(self.root, file)))

        if self.transform is not None:
            image = self.transform(image=image)['image']

        if 'dog' in file:
            label = 1
        elif 'cat' in file:
            label = 0
        else:
            label = -1
        return image, label
