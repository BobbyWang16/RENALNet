"""
    This file is for testing the model
    written by: Bobby Wang
"""
import pandas as pd
import numpy as np
import os
import torch
from util.setting import parse_opts
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score
from models.maskcnn import MaskAttention3DCNN
from util.dataset import get_kidney_loader
import warnings
warnings.filterwarnings("ignore")

def bootstrap_ci(metric_func, y_prob, y_true, n_iterations=1000, ci=0.95):
    """
    使用bootstrap方法计算指标的置信区间
    
    参数：
    metric_func: 计算指标的函数
    y_prob: 预测概率
    y_true: 真实标签
    n_iterations: bootstrap迭代次数
    ci: 置信区间水平
    
    返回：
    tuple: (下界, 上界)
    """
    scores = []
    size = len(y_true)
    
    for _ in range(n_iterations):
        # 随机抽样
        indices = np.random.randint(0, size, size)
        score = metric_func(y_true[indices], y_prob[indices])
        scores.append(score)
    
    # 计算置信区间
    lower = np.percentile(scores, ((1-ci)/2)*100)
    upper = np.percentile(scores, (1-(1-ci)/2)*100)
    
    return round(lower, 3), round(upper, 3)

def calculate_metrics(y_prob, y_true):
    """
    计算不平衡二分类问题的评价指标，包括95%置信区间
    
    参数：
    y_prob: numpy array, 预测为正类(少数类)的概率值
    y_true: numpy array, 真实标签 (0为多数类，1为少数类)
    
    返回：
    dict: 包含评价指标和置信区间的字典
    """
    # 1. AUC
    auc_score = roc_auc_score(y_true, y_prob)
    auc_ci = bootstrap_ci(roc_auc_score, y_prob, y_true)
    
    # 2. 通过优化F1-score选择最佳阈值
    # precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    # f1_scores = np.nan_to_num(f1_scores)
    # best_threshold = thresholds[np.argmax(f1_scores[:-1])]
    best_threshold = 0.5
    
    # 使用最佳阈值进行预测
    y_pred = (y_prob >= best_threshold).astype(int)
    
    # 3. F1-score
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_ci = bootstrap_ci(lambda y_t, y_p: f1_score(y_t, (y_p >= best_threshold).astype(int)),
                        y_prob, y_true)
    
    # 4. ACC
    acc = accuracy_score(y_true, y_pred)
    acc_ci = bootstrap_ci(lambda y_t, y_p: accuracy_score(y_t, (y_p >= best_threshold).astype(int)),
                         y_prob, y_true)
    
    # 5. MCC
    mcc = matthews_corrcoef(y_true, y_pred)
    mcc_ci = bootstrap_ci(lambda y_t, y_p: matthews_corrcoef(y_t, (y_p >= best_threshold).astype(int)),
                         y_prob, y_true)
    
    # 6. AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)
    
    def auprc_score(y_t, y_p):
        p, r, _ = precision_recall_curve(y_t, y_p)
        return auc(r, p)
    
    auprc_ci = bootstrap_ci(auprc_score, y_prob, y_true)
    
    # 7. Precision
    precisionscore = precision_score(y_true, y_pred)
    precision_ci = bootstrap_ci(lambda y_t, y_p: precision_score(y_t, (y_p >= best_threshold).astype(int)),
                              y_prob, y_true)

    # 8. Recall
    recallscore = recall_score(y_true, y_pred)
    recall_ci = bootstrap_ci(lambda y_t, y_p: recall_score(y_t, (y_p >= best_threshold).astype(int)),
                            y_prob, y_true)

    metrics = {
        'AUC': round(auc_score, 3),
        'AUC_CI': auc_ci,
        'F1': round(f1, 3),
        'F1_CI': f1_ci,
        'ACC': round(acc, 3),
        'ACC_CI': acc_ci,
        'MCC': round(mcc, 3),
        'MCC_CI': mcc_ci,
        'AUPRC': round(auprc, 3),
        'AUPRC_CI': auprc_ci,
        # 'threshold': round(best_threshold, 3),
        'Precision': round(precisionscore, 3),
        'Precision_CI': precision_ci,
        'Recall': round(recallscore, 3),
        'Recall_CI': recall_ci
    }
    return metrics

def format_metrics(metrics):
    """
    格式化指标输出，包含置信区间
    """
    formatted = {}
    for key in metrics:
        if key.endswith('_CI'):
            continue
        if key + '_CI' in metrics:
            formatted[key] = f"{metrics[key]} ({metrics[key+'_CI'][0]}-{metrics[key+'_CI'][1]})"
        else:
            formatted[key] = f"{metrics[key]}"
    
    return formatted

def merge_dicts_to_df(*dicts):
    """
    将多个评估指标字典合并为一个DataFrame
    
    参数:
    *dicts: 多个包含评估指标的字典
    
    返回:
    merged_df: 合并后的DataFrame
    """
    # 创建空的DataFrame保存所有结果
    merged_df = pd.DataFrame()
    
    # 创建索引列表
    new_index = ['train', 'inter_test', 'henan_test', 'kits_test']
    
    # 遍历所有字典
    for i, d in enumerate(dicts, 1):
        # 提取主要指标
        metrics = {
            'AUC': d['AUC'],
            'AUC_CI': f"({d['AUC_CI'][0]:.3f}-{d['AUC_CI'][1]:.3f})",
            'AUPRC': d['AUPRC'],
            'AUPRC_CI': f"({d['AUPRC_CI'][0]:.3f}-{d['AUPRC_CI'][1]:.3f})",
            'F1': d['F1'],
            'F1_CI': f"({d['F1_CI'][0]:.3f}-{d['F1_CI'][1]:.3f})",
            'ACC': d['ACC'],
            'ACC_CI': f"({d['ACC_CI'][0]:.3f}-{d['ACC_CI'][1]:.3f})",
            'MCC': d['MCC'],
            'MCC_CI': f"({d['MCC_CI'][0]:.3f}-{d['MCC_CI'][1]:.3f})",
            'Precision': d['Precision'],
            'Precision_CI': f"({d['Precision_CI'][0]:.3f}-{d['Precision_CI'][1]:.3f})",
            'Recall': d['Recall'],
            'Recall_CI': f"({d['Recall_CI'][0]:.3f}-{d['Recall_CI'][1]:.3f})",
        }
        
        # 转换为DataFrame并添加模型标识
        df = pd.DataFrame([metrics])
        
        # 合并到主DataFrame
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    
    # 重排列columns
    column_order = ['AUC', 'AUC_CI', 'AUPRC', 'AUPRC_CI', 
                    'F1', 'F1_CI', 'ACC', 'ACC_CI', 'MCC', 'MCC_CI',
                    'Precision', 'Precision_CI', 'Recall', 'Recall_CI']
    merged_df = merged_df[column_order]
    
    # 添加索引列
    merged_df.insert(0, 'Dataset', new_index)
    
    return merged_df

def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    all_predictions = []
    all_targets = [] 
    namelist = []
    
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
            namelist.extend(name)
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets) 
    # print(all_predictions, all_targets)
    namelist = [name.split('&')[1] for name in namelist]
    result = calculate_metrics(all_predictions, all_targets)
    return result, all_predictions, all_targets, namelist

def main():
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
    # 配置参数
    sets = parse_opts()
    sets.batch_size = 16
    sets.phase = 'train_test'
    sets.stage = 'train'
    train_loader = get_kidney_loader(
        sets, 
        batch_size=sets.batch_size, 
        do_augment=False
    )
    sets.stage = 'test'
    # 获取数据加载器
    test_loader = get_kidney_loader(
        sets, 
        batch_size=sets.batch_size, 
        do_augment=False
    )
    sets.phase = 'kits_test'
    kits_test_loader = get_kidney_loader(
        sets,
        batch_size=sets.batch_size,
        do_augment=False
    )
    sets.phase = 'henan_test'
    henan_test_loader = get_kidney_loader(
        sets,
        batch_size=sets.batch_size,
        do_augment=False
    )
    
    # 加载模型
    # checkpoint_path = './checkpoint/chk_1/kits_epoch71.pth'
    checkpoint_path = './checkpoint/chk_1/kits_epoch81.pth'
    model = MaskAttention3DCNN()
    model = model.cuda()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    
    # 测试模型
    result_train, train_proba, y_train, train_index = test_model(model, train_loader)
    result_val, val_proba, y_val, val_index = test_model(model, test_loader)

    # kits, henan
    result_kits, kits_proba, y_kits, kits_index = test_model(model, kits_test_loader)
    result_henan, henan_proba, y_henan, henan_index = test_model(model, henan_test_loader)


    # 将计算指标转为表格
    train_df = pd.DataFrame({
        'dataset': 'train',
        'name': train_index,
        'ground_truth': y_train,
        'probability': train_proba
    })

    val_df = pd.DataFrame({
        'dataset': 'val', 
        'name': val_index,
        'ground_truth': y_val,
        'probability': val_proba
    })

    henan_df = pd.DataFrame({
        'dataset': 'henan',
        'name': henan_index,
        'ground_truth': y_henan,
        'probability': henan_proba
    })

    kits_df = pd.DataFrame({
        'dataset': 'kits',
        'name': kits_index,
        'ground_truth': y_kits,
        'probability': kits_proba
    })
    # 合并所有DataFrame
    final_df = pd.concat([train_df, val_df, henan_df, kits_df], ignore_index=True)
    # 保存为Excel文件
    final_df.to_excel('dl.xlsx', index=False)

    # print(result_val)
    # print(result_henan)
    # print(result_kits)

    # 合并字典
    merged_results = merge_dicts_to_df(result_train, result_val, result_henan, result_kits)

    # 保存为CSV文件
    merged_results.to_excel('results_dl.xlsx', index=False)

if __name__ == '__main__':

    main()