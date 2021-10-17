# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim, true_divide