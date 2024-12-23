"""
    this file is dataset for kidney tumor classification model
    written by bobby Wang
"""

import math
import numpy as np
import pandas as pd
import os
import random
import torch
from torch.utils.data import Dataset
from util.setting import parse_opts
# import nibable as nib
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
from skimage import measure
import torchvision.transforms as transforms

def adjust_window(ct_image, window_width, window_level):
    """
    Adjusts the window level and window width of an image.
    """
    win_min = window_level - (window_width // 2)
    win_max = window_level + (window_width // 2)
    image_windowed = np.clip(ct_image, win_min, win_max)
    image_windowed = 255 * (image_windowed - win_min) / window_width
    image_windowed = np.float32(image_windowed)
    return image_windowed


class KidneyTumorDataset(Dataset):
    def __init__(self, sets):
        self.gt_path = sets.gt_path
        self.image_path = sets.image_path
        self.mask_path = sets.mask_path
        self.input_size = sets.input_size
        self.phase = sets.phase
        self.stage = sets.stage

        clinical_data = pd.read_csv(self.gt_path)
        if self.phase == 'train_test':
            X_train, X_test, y_train, y_test = train_test_split(clinical_data['name'], clinical_data['class'], test_size=0.3, random_state=42)
            
            if self.stage == 'test':
                self.img_list = X_test.values
                self.label_list = y_test.values
            elif self.stage == 'train':
                self.img_list = X_train.values
                self.label_list = y_train.values
                # self.img_list = pd.concat([X_train, X_test[:30]]).values
                # self.label_list = pd.concat([y_train, y_test[:30]]).values
            else:
                raise ValueError('stage must be train or test')
        else:

            self.img_list = clinical_data['name'].values
            self.label_list = clinical_data['class'].values

        self.indices = [i for i in range(len(self.img_list)) if self.label_list[i] == 1]

    def __len__(self):
            return len(self.img_list)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.image_path, self.phase, self.img_list[idx]+".npy"))
        img = adjust_window(img, 600, 40)
        new_img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        mask = np.load(os.path.join(self.mask_path, self.phase, self.img_list[idx]+".npy"))
        # mask[mask == 3] = 1
        mask[mask != 2] = 0
        mask[mask == 2] = 1
        label = self.label_list[idx].astype(np.float32)
        return new_img, mask, label


class KidneyTumorDataset_balanced(Dataset):
    def __init__(self, sets, sampling_strategy='balanced'):
        self.gt_path = sets.gt_path
        self.image_path = sets.image_path
        self.sampling_strategy = sampling_strategy
        self.phase = sets.phase
        self.stage = sets.stage
        
        # 读取数据
        clinical_data = pd.read_csv(self.gt_path)
        
        if self.phase == 'train_test':
            X_train, X_test, y_train, y_test = train_test_split(
                clinical_data['name'], 
                clinical_data['class'], 
                test_size=0.3, 
                stratify=clinical_data['class'],  # 确保划分时保持类别比例
                random_state=42
            )

            
            if self.stage == 'train':
                self.img_list = X_train.values
                self.label_list = y_train.values
            else:
                self.img_list = X_test.values
                self.label_list = y_test.values
        
        # 获取正负样本索引
        self.positive_indices = np.where(self.label_list == 1)[0]
        self.negative_indices = np.where(self.label_list == 0)[0]
        
        # 根据采样策略处理数据
        if self.stage == 'train':
            self.process_sampling_strategy()

    def process_sampling_strategy(self):
        if self.sampling_strategy == 'balanced':
            # 平衡采样：随机采样负样本使其数量与正样本相等
            selected_negative = np.random.choice(
                self.negative_indices,
                size=len(self.positive_indices),
                replace=False
            )
            selected_indices = np.concatenate([self.positive_indices, selected_negative])
            
        elif self.sampling_strategy == 'oversample':
            # 过采样：对少数类进行重复采样
            oversampled_positive = np.random.choice(
                self.positive_indices,
                size=len(self.negative_indices) - len(self.positive_indices),
                replace=True
            )
            selected_indices = np.concatenate([
                self.positive_indices,
                oversampled_positive,
                self.negative_indices
            ])
            
        elif self.sampling_strategy == 'weighted':
            # 加权采样：保持原始数据，但在DataLoader中使用权重
            selected_indices = np.concatenate([self.positive_indices, self.negative_indices])
            # 计算采样权重
            self.weights = np.ones(len(selected_indices))
            self.weights[self.positive_indices] = len(self.negative_indices) / len(self.positive_indices)
            
        else:  # 'none'
            # 保持原始数据不变
            selected_indices = np.concatenate([self.positive_indices, self.negative_indices])
        
        # 更新数据列表
        self.img_list = self.img_list[selected_indices]
        self.label_list = self.label_list[selected_indices]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.image_path, self.phase, self.img_list[idx]+".npy"))
        img = adjust_window(img, 400, 40)
        new_img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        label = self.label_list[idx].astype(np.float32)
        return new_img, label

if __name__ == "__main__":
    # root_dir = '/home/fbz/data/wch/kidney_tumor_classification/data/new_data'
    # data_path = '/home/fbz/data/wch/kidney_tumor_classification/data/clinical_feature.xlsx'
    sets = parse_opts()
    sets.stage = 'train'

    # excel_path = sets.excel_path
    # clinical_data = pd.read_excel(excel_path)
    # names = clinical_data['name']
    # labels = clinical_data['type']
    # X_train, X_test, y_train, y_test = train_test_split(names, labels, test_size=0.3, random_state=42)
    train_dataset = KidneyTumorDataset(sets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=8)
    for i, (img,mask, label) in enumerate(train_loader):
        print(img.shape, mask.shape, label.shape)


