# 文件名: src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from tqdm import tqdm
import os

class Trainer:
    """统一的训练器类"""
    def __init__(self, model, config):
        self.device = torch.device(config['device'])
        self.model = model.to(self.device)
        train_config = config['training_config']
        
        self.epochs = config['epochs']
        self.patience = train_config['patience']
        self.model_save_path = os.path.join(config['model_dir'], f"{config['model_type']}_best.pth")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience // 2, factor=0.5)
        self.criterion = nn.MSELoss()
        self.early_stopping = EarlyStopping(patience=self.patience)
        self.checkpoint = ModelCheckpoint(save_path=self.model_save_path)

        print(f"✅ 训练器初始化完成，使用设备: {self.device}")

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Train]", unit="batch")
        for data, target in progress_bar:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(data)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(val_loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Valid]", unit="batch", leave=False)
        with torch.no_grad():
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                pred = self.model(data)
                loss = self.criterion(pred, target)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.6f}")
        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader):
        train_losses, val_losses = [], []
        print(f"\n{'='*25} 开始训练 {'='*25}")
        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch {self.current_epoch} 总结 | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            if self.checkpoint(self.model, val_loss): print(f"  💾 模型检查点已保存 (Val Loss: {val_loss:.6f})")
            if self.early_stopping(val_loss):
                print(f"  🛑 早停机制已触发，训练在 Epoch {self.current_epoch} 终止。")
                break
        print(f"\n{'='*25} 训练完成 {'='*25}")
        return train_losses, val_losses

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience, self.min_delta, self.counter, self.best_loss = patience, min_delta, 0, float('inf')
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.counter = val_loss, 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class ModelCheckpoint:
    def __init__(self, save_path, monitor='val_loss', mode='min'):
        self.save_path, self.monitor, self.mode = save_path, monitor, mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    def __call__(self, model, value):
        is_better = (self.mode == 'min' and value < self.best_value) or (self.mode == 'max' and value > self.best_value)
        if is_better:
            self.best_value = value
            torch.save(model.state_dict(), self.save_path)
        return is_better