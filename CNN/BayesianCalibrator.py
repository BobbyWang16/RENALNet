import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import deque

class EnhancedBayesianCalibrator:
    def __init__(self,
                 train_pos_rate=0.1,
                 test_pos_rate=0.2,
                 n_neighbors=5,
                 smooth_factor=0.3,
                 memory_size=1000,
                 bootstrap_ratio=0.8,
                 n_bootstraps=5):
        # 贝叶斯校准参数
        self.train_pos_rate = train_pos_rate
        self.test_pos_rate = test_pos_rate
        self.pos_ratio = test_pos_rate / train_pos_rate
        self.neg_ratio = (1 - test_pos_rate) / (1 - train_pos_rate)
        
        # AUC增强参数
        self.n_neighbors = n_neighbors
        self.smooth_factor = smooth_factor
        self.memory_size = memory_size
        self.bootstrap_ratio = bootstrap_ratio
        self.n_bootstraps = n_bootstraps
        
        # 记忆库
        self.feature_memory = deque(maxlen=memory_size)
        self.pred_memory = deque(maxlen=memory_size)
        
    def _bayesian_calibrate(self, pred_prob):
        """贝叶斯校准"""
        numerator = pred_prob * self.pos_ratio
        denominator = pred_prob * self.pos_ratio + (1 - pred_prob) * self.neg_ratio
        return numerator / denominator
    
    def _neighborhood_smooth(self, X, predictions):
        """局部平滑"""
        if len(self.feature_memory) < self.n_neighbors:
            return predictions
            
        memory_features = np.array(list(self.feature_memory))
        memory_preds = np.array(list(self.pred_memory))
        
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors)
        nbrs.fit(memory_features)
        distances, indices = nbrs.kneighbors(X)
        
        weights = np.exp(-distances / distances.mean())
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        smooth_predictions = np.zeros_like(predictions)
        for i in range(len(predictions)):
            neighbor_preds = memory_preds[indices[i]]
            smooth_predictions[i] = (1 - self.smooth_factor) * predictions[i] + \
                                  self.smooth_factor * np.sum(weights[i] * neighbor_preds)
        
        return smooth_predictions
    
    def _bootstrap_ensemble(self, X, predictions):
        """自举集成"""
        if len(self.feature_memory) < 10:
            return predictions
            
        ensemble_preds = []
        memory_features = np.array(list(self.feature_memory))
        memory_preds = np.array(list(self.pred_memory))
        
        for _ in range(self.n_bootstraps):
            bootstrap_idx = np.random.choice(
                len(memory_features),
                size=int(len(memory_features) * self.bootstrap_ratio),
                replace=True
            )
            
            boot_features = memory_features[bootstrap_idx]
            boot_preds = memory_preds[bootstrap_idx]
            
            nbrs = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(boot_features)))
            nbrs.fit(boot_features)
            distances, indices = nbrs.kneighbors(X)
            
            weights = np.exp(-distances / distances.mean())
            weights = weights / weights.sum(axis=1, keepdims=True)
            
            bootstrap_predictions = np.zeros_like(predictions)
            for i in range(len(predictions)):
                neighbor_preds = boot_preds[indices[i]]
                bootstrap_predictions[i] = np.sum(weights[i] * neighbor_preds)
                
            ensemble_preds.append(bootstrap_predictions)
            
        ensemble_predictions = np.mean(ensemble_preds, axis=0)
        return 0.7 * predictions + 0.3 * ensemble_predictions
    
    def predict(self, X, raw_predictions, with_uncertainty=False, uncertainty=0.1):
        """主预测函数"""
        # 1. 贝叶斯校准
        if with_uncertainty:
            lower_rate = self.test_pos_rate * (1 - uncertainty)
            upper_rate = min(self.test_pos_rate * (1 + uncertainty), 1.0)
            
            lower_calibrator = EnhancedBayesianCalibrator(self.train_pos_rate, lower_rate)
            upper_calibrator = EnhancedBayesianCalibrator(self.train_pos_rate, upper_rate)
            
            calibrated_lower = lower_calibrator._bayesian_calibrate(raw_predictions)
            calibrated_upper = upper_calibrator._bayesian_calibrate(raw_predictions)
            calibrated_predictions = (calibrated_lower + calibrated_upper) / 2
        else:
            calibrated_predictions = self._bayesian_calibrate(raw_predictions)
        
        # 2. AUC增强
        smooth_predictions = self._neighborhood_smooth(X, calibrated_predictions)
        enhanced_predictions = self._bootstrap_ensemble(X, smooth_predictions)
        
        # 3. 更新记忆库
        for x, p in zip(X, enhanced_predictions):
            self.feature_memory.append(x)
            self.pred_memory.append(p)
        
        if with_uncertainty:
            return enhanced_predictions, (calibrated_lower, calibrated_upper)
        return enhanced_predictions

# 使用示例
def enhanced_inference(model, X_test, train_pos_rate=0.1, test_pos_rate=0.2):
    # 初始化增强型校准器
    calibrator = EnhancedBayesianCalibrator(
        train_pos_rate=train_pos_rate,
        test_pos_rate=test_pos_rate,
        n_neighbors=5,
        smooth_factor=0.3,
        memory_size=1000,
        bootstrap_ratio=0.8,
        n_bootstraps=5
    )
    
    batch_size = 32
    all_predictions = []
    
    # 分批处理
    for i in range(0, len(X_test), batch_size):
        batch_X = X_test[i:i + batch_size]
        raw_predictions = model(batch_X)
        
        # 带不确定性的预测
        enhanced_predictions, (lower_bound, upper_bound) = calibrator.predict(
            batch_X,
            raw_predictions,
            with_uncertainty=True,
            uncertainty=0.1
        )
        
        all_predictions.extend(enhanced_predictions)
    
    return np.array(all_predictions)

# 评估示例
def evaluate_enhancement(model, X_test, y_test):
    from sklearn.metrics import roc_auc_score
    
    # 原始预测
    raw_predictions = model(X_test)
    raw_auc = roc_auc_score(y_test, raw_predictions)
    
    # 增强预测
    enhanced_predictions = enhanced_inference(model, X_test)
    enhanced_auc = roc_auc_score(y_test, enhanced_predictions)
    
    print(f"Raw AUC: {raw_auc:.4f}")
    print(f"Enhanced AUC: {enhanced_auc:.4f}")
    print(f"Improvement: {(enhanced_auc-raw_auc)*100:.2f}%")
    
    return raw_predictions, enhanced_predictions