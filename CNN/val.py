"""
    This file is for testing the model
    written by: Bobby Wang
"""
import pandas as pd
import numpy as np
import os
import torch
from util.setting import parse_opts
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from models.maskcnn import MaskAttention3DCNN
from util.dataset import get_kidney_loader
from multiprocessing import freeze_support

def test_model(model, test_loader):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_predictions = []
    all_targets = []
    all_names = []  # 用于存储样本名称
    
    with torch.no_grad():
        for data, mask, target, names in test_loader:  # 假设dataloader现在也返回names
            data = data.to(device)
            mask = mask.to(device)
            target = target.to(device)
            
            outputs = model(data, mask)
            predictions = torch.sigmoid(outputs)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_names.extend(names)  # 添加名称
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'Name': all_names,
        'Predicted_Probability': all_predictions.flatten(),
        'Ground_Truth': all_targets.flatten()
    })
    
    # 保存到Excel
    # output_path = './results/predictions_.xlsx'
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # results_df.to_excel(output_path, index=False)
    # print(f"Results saved to {output_path}")
    
    try:
        auc_score = roc_auc_score(all_targets, all_predictions)
        return auc_score, results_df
    except Exception as e:
        print(f"Error in calculating AUC: {e}")
        return 0.0, results_df

def main():
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 配置参数
    sets = parse_opts()
    sets.phase = 'train_test'
    sets.stage = 'test'
    sets.batch_size = 16
    
    # 获取数据加载器
    test_loader = get_kidney_loader(
        sets, 
        batch_size=sets.batch_size, 
        do_augment=False,
        num_workers=2
    )
    sets.phase = 'kits_test'
    kits_test_loader = get_kidney_loader(
        sets,
        batch_size=sets.batch_size,
        do_augment=False,
        num_workers=2
    )
    sets.phase = 'henan_test'
    henan_test_loader = get_kidney_loader(
        sets,
        batch_size=sets.batch_size,
        do_augment=False,
        num_workers=2
    )
    
    # 加载模型
    checkpoint_path = './checkpoint/chk_5/kits_epoch70.pth'
    model = MaskAttention3DCNN()
    model = model.cuda()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    
    # 测试模型
    auc_final, results_df = test_model(model, test_loader)
    print("auc={}.".format(auc_final))
    
    # # 添加额外的统计信息
    # print("\nPrediction Statistics:")
    # print("Total samples:", len(results_df))
    # print("Positive predictions (>0.5):", len(results_df[results_df['Predicted_Probability'] > 0.5]))
    # print("Actual positive cases:", len(results_df[results_df['Ground_Truth'] == 1]))

    # kits
    auc_final_kits, results_df = test_model(model, kits_test_loader)
    print("auc={}.".format(auc_final_kits))

    # henan
    auc_final_henan, results_df = test_model(model, henan_test_loader)
    print("auc={}.".format(auc_final_henan))

if __name__ == '__main__':
    freeze_support()
    main()