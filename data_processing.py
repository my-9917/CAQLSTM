import wfdb
import numpy as np
import os
from scipy.signal import resample
from tqdm import tqdm
import config
from src.utils.chaos_features import calculate_chaos_features

def process_data():
    # 获取配置
    record_name = config.DATASET_CONFIG['record_name']
    seq_len = config.DATASET_CONFIG['sequence_length']
    train_split = config.DATASET_CONFIG['train_split']
    output_dir = config.DATA_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_file = os.path.join(output_dir, f'train_{record_name}_chaos.npy')
    val_file = os.path.join(output_dir, f'val_{record_name}_chaos.npy')
    if os.path.exists(train_file) and os.path.exists(val_file):
        print("✅ 带混沌特征的数据文件已存在，跳过处理。")
        return
    # 读取本地信号
    print(f"📖 正在从 data/raw/{record_name} 读取数据...")
    record_path = os.path.join("data/raw", record_name)
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]
    signal = (signal - np.mean(signal)) / np.std(signal)
    # 切分序列并提取混沌特征
    print("🧠 正在创建序列并计算混沌特征...")
    X_raw, X_chaos, y = [], [], []
    max_sequences = config.DATA_CONFIG['max_sequences']
    print(f"📊 最大序列数限制: {max_sequences}")
    
    # 计算实际要处理的序列数
    total_possible_sequences = len(signal) - seq_len
    actual_sequences_to_process = min(max_sequences, total_possible_sequences)
    print(f"📈 总可能序列数: {total_possible_sequences}, 实际处理数: {actual_sequences_to_process}")
    
    for i in tqdm(range(actual_sequences_to_process), desc="Processing sequences"):
        raw_seq = signal[i:i+seq_len]
        X_raw.append(raw_seq)
        chaos_feats = calculate_chaos_features(raw_seq)
        X_chaos.append(chaos_feats)
        y.append(signal[i+seq_len])
    
    print(f"📈 实际处理的序列数: {len(X_raw)}")
    X_raw = np.array(X_raw, dtype=np.float32)
    X_chaos = np.array(X_chaos, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    X_raw = np.expand_dims(X_raw, axis=-1)
    y = np.expand_dims(y, axis=-1)
    # 划分训练/验证集
    split_idx = int(len(X_raw) * train_split)
    X_train_raw, X_val_raw = X_raw[:split_idx], X_raw[split_idx:]
    X_train_chaos, X_val_chaos = X_chaos[:split_idx], X_chaos[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    # 保存数据
    np.save(train_file, {'X_raw': X_train_raw, 'X_chaos': X_train_chaos, 'y': y_train})
    np.save(val_file, {'X_raw': X_val_raw, 'X_chaos': X_val_chaos, 'y': y_val})
    print(f"✅ 数据处理完成，已保存至 {output_dir}")
    print(f"   训练集大小: {X_train_raw.shape}, 混沌特征: {X_train_chaos.shape}")
    print(f"   验证集大小: {X_val_raw.shape}, 混沌特征: {X_val_chaos.shape}")

if __name__ == '__main__':
    process_data()