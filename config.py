import torch
import os

# ============================================================================
# 模式总开关
# ============================================================================
QUICK_TEST_MODE = False  # True: 快速调试模式, False: 完整训练模式

# ============================================================================
# 混沌特征配置
# ============================================================================
CHAOS_CONFIG = {
    "num_features": 3, # 与 calculate_chaos_features 函数返回的特征数量一致
}

# ============================================================================
# 通用配置
# ============================================================================
DATASET_CONFIG = {
    "record_name": "16539",    # PhysioNet nsrdb 记录名
    "sequence_length": 64,     # 输入序列长度
    "train_split": 0.8,        # 训练集比例
}

# 根据模式定义不同配置
if QUICK_TEST_MODE:
    print("🚀 已启用快速测试模式")
    MODEL_TYPE = "qlstm"
    MAX_SEQUENCES = 500
    EPOCHS = 5
    BATCH_SIZE = 8
    QLSTM_CONFIG = {
        "hidden_dim": 32,
        "n_qubits": 2,
        "n_qlayers": 1,
        "lstm_layers": 1,
    }
    TRAINING_CONFIG = {
        "learning_rate": 0.005,
        "weight_decay": 1e-4,
        "patience": 5,
    }
else:
    print("🏆 已启用完整训练模式")
    MODEL_TYPE = "qlstm"
    MAX_SEQUENCES = 20000
    EPOCHS = 50
    BATCH_SIZE = 32
    QLSTM_CONFIG = {
        "hidden_dim": 32,
        "n_qubits": 4,
        "n_qlayers": 2,
        "lstm_layers": 2,
    }
    TRAINING_CONFIG = {
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "patience": 10,
    }

# ------------------------------------------------------------------------
# 数据配置
# ------------------------------------------------------------------------
DATA_CONFIG = {
    "max_sequences": 50000,  #data_processing的数据范围
    "batch_size": BATCH_SIZE,
    "shuffle_train": True,
    "shuffle_val": False,
    "drop_last": True,
}

# ============================================================================
# 路径与设备
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/processed"
RESULTS_DIR = "results"
MODEL_DIR = "models"

# 动态生成带混沌特征的数据文件路径
TRAIN_DATA_PATH = os.path.join(DATA_DIR, f"train_{DATASET_CONFIG['record_name']}_chaos.npy")
VAL_DATA_PATH = os.path.join(DATA_DIR, f"val_{DATASET_CONFIG['record_name']}_chaos.npy")
