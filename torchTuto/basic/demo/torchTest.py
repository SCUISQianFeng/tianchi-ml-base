# -*- coding:utf-8 -*-

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
