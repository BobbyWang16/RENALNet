"""
    This file is for training the model
    written by: Bobby Wang
"""

import os
import logging
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from util.setting import parse_opts
from torch.utils.data import DataLoader
from models.model import generate_model
from models.medicalnet import Medical3DCNN
from models.maskcnn import MaskAttention3DCNN
from util.dataloader import KidneyTumorDataset, KidneyTumorDataset_balanced
from util.dataset import get_kidney_loader

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('train.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def train_model(model, train_loader, test_loader, sets):
    optimizer = optim.Adam(model.parameters(), lr=sets.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_epochs = sets.n_epochs
    
    # 损失函数 (二分类)
    criterion = FocalLoss(alpha=0.8, gamma=2.0)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, mask, target, name) in enumerate(train_loader):
            
            # 将数据转移到device上
            data = data.to(device)
            mask = mask.to(device)
            # 不需要view，因为现在target是类别索引
            target = target.to(device)

            # 前向传播
            predict = model(data, mask)
            loss = criterion(predict, target)
            loss_sum = loss.sum()
            loss_mean = loss.mean()
            total_loss += loss_sum.item()

            # 反向传播
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

        # 更新学习率
        scheduler.step()
        # 计算测试集上的auc
        model.eval()
        test_auc = test_model(model, test_loader)
        train_auc = test_model(model, train_loader)
        model.train()

        # 打印训练信息
        print(f'Train Epoch: [{epoch+1} / {num_epochs}], Step: [{batch_idx+1} / {len(train_loader)}], '
              f'Total_Loss: {total_loss}, LR: {scheduler.get_last_lr()}, Test_AUC: {test_auc}, Train_auc: {train_auc}')
        
        # 每隔1轮保存一次模型
        if (epoch+1) % sets.save_intervals == 0:
            if not os.path.exists(sets.save_folder):
                os.makedirs(sets.save_folder)
            logger.info(f'Saving model at epoch {epoch+1}')
            torch.save(model.state_dict(), sets.save_folder + f'/kits_epoch{epoch+1}.pth')

def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    all_predictions = []
    all_targets = [] 
    
    with torch.no_grad():
        for data, mask, target, name in test_loader:
            data = data.to(device)
            mask = mask.to(device)
            target = target.to(device)
            outputs = model(data, mask)
            
            # 对两个输出使用softmax得到概率
            probabilities = F.softmax(outputs, dim=1)
            # 取第二个类别（正类）的概率
            positive_probs = probabilities[:, 1]
            
            all_predictions.extend(positive_probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets) 
    
    try:
        auc_score = roc_auc_score(all_targets, all_predictions)
        return auc_score
    except Exception as e:
        print(f"Error in calculating AUC: {e}")
        return 0.0

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出，shape为(N, 2)，表示每个样本属于两个类别的logits
            targets: 真实标签，shape为(N,)，值为0或1
        """
        # 将targets转换为one-hot编码
        targets_one_hot = F.one_hot(targets.long(), num_classes=2).float()
        
        # 计算softmax
        inputs_softmax = F.softmax(inputs, dim=1)
        
        # 计算交叉熵
        ce_loss = -targets_one_hot * torch.log(inputs_softmax + 1e-10)
        
        # 计算focal loss的调制因子
        pt = torch.sum(targets_one_hot * inputs_softmax, dim=1)
        focal_weight = (1 - pt) ** self.gamma
        
        # 添加alpha权重
        alpha_weight = torch.where(targets == 1, 
                                 self.alpha * torch.ones_like(targets),
                                 (1 - self.alpha) * torch.ones_like(targets))
        
        # 计算最终的focal loss
        focal_loss = alpha_weight * focal_weight * torch.sum(ce_loss, dim=1)
        
        return focal_loss

def set_seed(seed=42):
    """设置随机种子以确保结果的可重复性"""
    random.seed(seed)  # Python的random模块
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # 多GPU
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = True  # True的话会自动寻找最适合当前配置的高效算法，设置为False可以保证实验结果的可重复性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子

def main():
    """
        Main function
    """
    # 在代码开头调用此函数
    # set_seed(42)  # 42可以换成任意整数

    # 设置CUDA_VISIBLE_DEVICES环境变量，只使用第一个GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    sets = parse_opts()
    # 训练
    sets.stage = 'train'
    sets.batch_size = 16
    sets.n_epochs = 100
    sets.learning_rate = 0.0001
    sets.save_intervals = 1
    sets.save_folder = "./checkpoint/chk_1"
    print(sets.save_folder)
    # train_dataset = KidneyTumorDataset(sets)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=8)
    train_loader = get_kidney_loader(sets, batch_size=sets.batch_size, do_augment=True)

    sets.stage = 'test'
    sets.batch_size = 32
    test_loader = get_kidney_loader(sets, batch_size=sets.batch_size, do_augment=False)
    # test_dataset = KidneyTumorDataset(sets)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=sets.batch_size, shuffle=False, num_workers=8)
   
    model = MaskAttention3DCNN()
    # 检查是否有可用的GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 将模型转移到device上
    model.to(device)

    train_model(model, train_loader, test_loader, sets)

    # 训练完成
    logger.info("Training finished!")

if __name__ == "__main__":
    main()