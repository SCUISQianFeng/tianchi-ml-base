# -*- coding:utf-8 -*-

import  os
import re
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class CatDog(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.tranform = transform
        self.images = os.listdir(root)
        self.images.sort(lambda x: int(re.findall(r'\d+', x)[0]))  #

    def __len__(self):
        return

    def __getitem__(self, index):