import nolds
import numpy as np
import warnings

# 忽略sklearn的UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._regression")

def calculate_chaos_features(sequence):
    # 计算单个时间序列窗口的混沌特征
    features = []
    if sequence.ndim > 1:
        sequence = sequence.flatten()
    # 样本熵
    try:
        sampen = nolds.sampen(sequence)
        features.append(sampen)
    except Exception:
        features.append(0.0)
    # DFA（赫斯特指数）
    try:
        hurst = nolds.dfa(sequence)
        features.append(hurst)
    except Exception:
        features.append(0.5)
    # 相关维数
    try:
        if len(sequence) > 10:
            corr_dim = nolds.corr_dim(sequence, emb_dim=2)
        else:
            corr_dim = 0.0
        features.append(corr_dim)
    except Exception:
        features.append(0.0)
    return np.array(features, dtype=np.float32)