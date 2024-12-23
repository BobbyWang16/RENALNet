"""
this is the 5-fold cross validation program
written by: Bobby Wang
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms, ToTensor
from sklearn.model_selection import KFold

#