"""
    This file is for testing the model
    written by: Bobby Wang
"""
import pandas as pd
import numpy as np
import os
import torch

from setting import parse_opts
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from models.model import generate_model
from util.dataloader import KidneyTumorDataset

sets = parse_opts()
sets.phase = 'test'
test_dataset = KidneyTumorDataset(sets)
test_loader = DataLoader(test_dataset, batch_size=sets.batch_size, shuffle=False, num_workers=8)

# 设置CUDA_VISIBLE_DEVICES环境变量，只使用第一个GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
checkpoint_path = './checkpoint/resnet_18/chk_3/kits_epoch22.pth'
model = generate_model(sets.model_depth)
nodel = model.cuda()
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# 预测
def evaluate(data_loader, model):
    pred = []
    label = []
    output = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.cuda()
            outputs = model(images)
            out = outputs.cpu().numpy()
            new_out = np.squeeze(out).tolist()
            result = (outputs > 0.5).int().cpu().numpy() #输出的cutoff值（可修改）
            new_result = np.squeeze(result).tolist()
            new_label = np.squeeze(labels.int()).tolist()
            pred = pred+new_result
            label = label+new_label
            output = output+new_out
    return pred, label, output

pred1, label1, output1 = evaluate(test_loader, model)
print(pred1, "\n", label1, "\n", output1)

# out1 = precision_score(pred1, label1)
# out2 = recall_score(pred1, label1)
out3 = roc_auc_score(label1, output1)

print(out3)

# 查看错误分类例
# 加载数据
clinical_data = pd.read_excel(sets.excel_path)
names = clinical_data['name']
labels = clinical_data['type']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(names, labels, test_size=0.3, random_state=42)
test_list = X_test.values.tolist()
test_label = y_test.values.tolist()
for i in range(len(label1)):
    if label1[i] == 0 and pred1[i] == 1:
        print(test_list[i], "predict 0 to 1, prob_score is", output1[i])
    elif label1[i] == 1 and pred1[i] == 0:
        print(test_list[i], "predict 1 to 0, prob_score is", output1[i])