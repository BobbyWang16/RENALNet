import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import os
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate

def adjust_window(ct_image, window_width, window_level):
    """
    Adjusts the window level and window width of an image.
    """
    win_min = window_level - (window_width // 2)
    win_max = window_level + (window_width // 2)
    image_windowed = np.clip(ct_image, win_min, win_max)
    image_windowed = (image_windowed - win_min) / window_width
    image_windowed = np.float32(image_windowed)
    return image_windowed

class NonRepeatingBalancedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        if batch_size < 2:
            raise ValueError("Batch size must be at least 2 to accommodate both positive and negative samples")
            
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_indices = dataset.indices.copy()  # 正样本索引
        self.neg_indices = [i for i in range(len(dataset)) if i not in self.pos_indices]  # 负样本索引
        
        # 计算需要的batch数量
        # 每个batch至少放一个正样本，剩余位置放负样本
        # 我们需要确保所有样本都能被使用
        total_samples = len(self.pos_indices) + len(self.neg_indices)
        self.num_batches = (total_samples + batch_size - 1) // batch_size
        
    def __iter__(self):

        # 复制并打乱索引
        pos_indices = self.pos_indices.copy()
        neg_indices = self.neg_indices.copy()
        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)
        
        # 先分配每个batch的第一个位置给正样本
        # 直到用完所有正样本或填满所有batch
        batches = []
        pos_idx = 0
        
        for i in range(self.num_batches):
            if pos_idx < len(pos_indices):
                # 创建新batch，以正样本开始
                batch = [pos_indices[pos_idx]]
                pos_idx += 1
            else:
                # 如果正样本用完了，创建空batch
                batch = []
            batches.append(batch)
        
        # 继续分配剩余的正样本（如果还有）
        while pos_idx < len(pos_indices):
            # 找到当前最小的batch
            min_batch_idx = min(range(len(batches)), key=lambda x: len(batches[x]))
            if len(batches[min_batch_idx]) < self.batch_size:
                batches[min_batch_idx].append(pos_indices[pos_idx])
                pos_idx += 1
            else:
                break
                
        # 分配负样本
        neg_idx = 0
        while neg_idx < len(neg_indices):
            # 找到当前最小的batch
            min_batch_idx = min(range(len(batches)), key=lambda x: len(batches[x]))
            if len(batches[min_batch_idx]) < self.batch_size:
                batches[min_batch_idx].append(neg_indices[neg_idx])
                neg_idx += 1
            else:
                break
        
        # 打乱每个batch内部的顺序并yield
        for batch in batches:
            if len(batch) > 0:  # 只返回非空batch
                np.random.shuffle(batch)
                yield batch
    
    def __len__(self):
        return self.num_batches

class RatioBalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, pos_ratio=0.2):
        """
        Args:
            dataset: 数据集
            batch_size: batch大小
            pos_ratio: 每个batch中正样本的目标比例 (0到1之间)
        """
        if not (0 < pos_ratio < 1):
            raise ValueError("pos_ratio must be between 0 and 1")
            
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        
        self.pos_indices = dataset.indices  # 正样本索引
        self.neg_indices = [i for i in range(len(dataset)) if i not in self.pos_indices]  # 负样本索引
        
        # 计算每个batch中的正负样本数量
        self.pos_per_batch = max(1, round(batch_size * pos_ratio))
        self.neg_per_batch = batch_size - self.pos_per_batch
        
        # 由于负样本不能重复，batch数量受限于负样本数量
        self.num_batches = len(self.neg_indices) // self.neg_per_batch
        
        if self.num_batches == 0:
            raise ValueError(f"Not enough negative samples for a single batch. "
                           f"Need at least {self.neg_per_batch} negative samples.")
        
        print(f"Batch composition: {self.pos_per_batch} positive + {self.neg_per_batch} negative samples")
        print(f"Total batches per epoch: {self.num_batches}")
        
    def __iter__(self):

        # 复制并打乱负样本索引
        neg_indices = self.neg_indices.copy()
        np.random.shuffle(neg_indices)
        
        # 创建足够长的正样本索引列表（需要时会重复）
        needed_pos = self.num_batches * self.pos_per_batch
        pos_indices_extended = []
        while len(pos_indices_extended) < needed_pos:
            pos_indices_temp = self.pos_indices.copy()
            np.random.shuffle(pos_indices_temp)
            pos_indices_extended.extend(pos_indices_temp)
        pos_indices_extended = pos_indices_extended[:needed_pos]
        
        # 按批次生成索引
        for i in range(self.num_batches):
            batch_indices = []
            
            # 添加正样本
            start_pos = i * self.pos_per_batch
            end_pos = start_pos + self.pos_per_batch
            batch_indices.extend(pos_indices_extended[start_pos:end_pos])
            
            # 添加负样本（顺序提取，不重复使用）
            start_neg = i * self.neg_per_batch
            end_neg = start_neg + self.neg_per_batch
            batch_indices.extend(neg_indices[start_neg:end_neg])
            
            # 打乱batch内的顺序
            np.random.shuffle(batch_indices)
            
            yield batch_indices
    
    def __len__(self):
        return self.num_batches

class KidneyAugmentor:
    def __init__(self, p_flip=0.5, p_rotate=0.3, rotate_angle_range=(-10, 10)):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.rotate_angle_range = rotate_angle_range
    
    def flip(self, image, mask):
        """左右翻转"""
        if np.random.random() < self.p_flip:
            image = np.flip(image, axis=-1).copy()
            mask = np.flip(mask, axis=-1).copy()
        return image, mask
    
    def rotate_3d(self, image, mask):
        """随机旋转"""
        if np.random.random() < self.p_rotate:
            angle = np.random.uniform(*self.rotate_angle_range)
            # 确保数组是连续的
            image = np.ascontiguousarray(image)
            mask = np.ascontiguousarray(mask)
            # 沿z轴旋转
            for z in range(image.shape[0]):
                image[z] = rotate(image[z], angle, reshape=False, mode='nearest')
                mask[z] = rotate(mask[z], angle, reshape=False, mode='nearest')
        return image, mask
    
    def __call__(self, image, mask):
        # 确保输入数组是连续的
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        
        image, mask = self.flip(image, mask)
        image, mask = self.rotate_3d(image, mask)
        return image, mask

class KidneyTumorDataset(Dataset):
    def __init__(self, sets, do_augment=False):
        self.gt_path = sets.gt_path
        self.image_path = sets.image_path
        self.mask_path = sets.mask_path
        self.input_size = sets.input_size
        self.phase = sets.phase
        self.stage = sets.stage
        self.do_augment = do_augment
        self.augmentor = KidneyAugmentor() if do_augment else None

        clinical_data = pd.read_excel(self.gt_path)
        clinical_data = clinical_data[clinical_data['exclusion'] == 1]
        if self.phase == 'train_test':
            clinical_label = clinical_data[(clinical_data['dataset'] == "tongji") | (clinical_data['dataset'] == "xiangyang")]

            X_train, X_test, y_train, y_test = train_test_split(
                clinical_label['dataset']+ '&' + clinical_label["name"], 
                clinical_label["class"], 
                test_size=0.3, 
                random_state=42, 
                stratify=clinical_label["class"]
            )

            if self.stage == 'test':
                self.img_list = X_test.values
                self.label_list = y_test.values
            elif self.stage == 'train':
                self.img_list = X_train.values
                self.label_list = y_train.values

            else:
                raise ValueError('stage must be train or test')
        elif self.phase == 'kits_test':
            clinical_label = clinical_data[clinical_data['dataset'] == "kits"]
            self.img_list = (clinical_label['dataset']+ '&' + clinical_label['name']).values
            self.label_list = clinical_label['class'].values
        elif self.phase == 'henan_test':
            clinical_label = clinical_data[clinical_data['dataset'] == "henan"]
            self.img_list = (clinical_label['dataset']+ '&' + clinical_label['name']).values
            self.label_list = clinical_label['class'].values
        else:
            raise ValueError('phase must be train_test, kits_test or henan_test')

        self.indices = [i for i in range(len(self.img_list)) if self.label_list[i] == 1]
        print(f"正样本数量: {len(self.indices)}")
        print(f"总样本数量: {len(self.img_list)}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # img = np.load(os.path.join(self.image_path, self.phase, self.img_list[idx]+".npy"))
        
        # mask = np.load(os.path.join(self.mask_path, self.phase, self.img_list[idx]+".npy"))
        img = np.load("./data/" + self.img_list[idx].replace("&", "/np_image_64/"))
        mask = np.load("./data/" + self.img_list[idx].replace("&", "/np_label_64/"))
        img = adjust_window(img, 400, 40)
        # mask[mask == 3] = 1
        mask[mask != 2] = 0
        mask[mask == 2] = 1
        # img = img.astype(np.float32)
        mask = mask.astype(np.float32) 
        # 确保数组是连续的
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)
        
        # 应用数据增强
        if self.do_augment and self.augmentor is not None:
            img, mask = self.augmentor(img, mask)
            
        new_img = np.ascontiguousarray(np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2])))
        label = self.label_list[idx].astype(np.float32)
        
        return new_img, mask, label, self.img_list[idx]

def get_kidney_loader(sets, batch_size=8, do_augment=True, num_workers=8):
    """创建数据加载器"""
    dataset = KidneyTumorDataset(sets, do_augment=do_augment)
    
    if sets.stage == 'train':
        # 训练时使用BalancedBatchSampler
        sampler = NonRepeatingBalancedSampler(dataset, batch_size)
        # sampler = RatioBalancedBatchSampler(dataset, batch_size, pos_ratio=0.25)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers
        )
    else:
        # 测试时使用普通的DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    return loader

# # 使用示例：
# if __name__ == '__main__':
#     from setting import parse_opts
#     sets = parse_opts()
#     sets.phase = 'train_test'
#     sets.stage = 'train'

#     train_loader = get_kidney_loader(
#         sets,
#         batch_size=16,
#         do_augment=True
#     )

#     for batch_idx, (data, mask, target) in enumerate(train_loader):
#         print(batch_idx, data.shape, mask.shape, target, sum(target))