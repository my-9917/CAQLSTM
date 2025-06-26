import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class ECGDataset(Dataset):
    def __init__(self, file_path, max_sequences=None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件未找到: {file_path}")
        data = np.load(file_path, allow_pickle=True).item()
        self.X_raw = torch.from_numpy(data['X_raw'])
        self.X_chaos = torch.from_numpy(data['X_chaos'])
        self.y = torch.from_numpy(data['y'])
        if max_sequences:
            print(f"⚠️  限制数据量: 使用前 {max_sequences} 个序列")
            self.X_raw = self.X_raw[:max_sequences]
            self.X_chaos = self.X_chaos[:max_sequences]
            self.y = self.y[:max_sequences]
    def __len__(self):
        return len(self.X_raw)
    def __getitem__(self, idx):
        x_raw = self.X_raw[idx]          # [seq_len, 1]
        x_chaos = self.X_chaos[idx]      # [num_chaos_features]
        seq_len = x_raw.shape[0]
        x_chaos_expanded = x_chaos.unsqueeze(0).expand(seq_len, -1) # [seq_len, num_chaos_features]
        x_combined = torch.cat([x_raw, x_chaos_expanded], dim=1)    # [seq_len, 1 + num_chaos_features]
        return x_combined, self.y[idx]
def create_data_loaders(train_path, val_path, batch_size, max_sequences):
    train_dataset = ECGDataset(train_path, max_sequences)
    val_dataset = ECGDataset(val_path, max_sequences)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader