import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import norm

def plot_training_curves(train_losses, val_losses, save_path):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='royalblue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='darkorange', linewidth=2)
    plt.title('Training & Validation Loss Curves', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 训练曲线图已保存至: {save_path}")

def plot_predictions(model, data_loader, device, save_path):
    """绘制部分预测值与真实值的对比图"""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            if len(all_preds) * data.size(0) > 500: # 获取足够样本后即可停止
                break
            
    preds = np.concatenate(all_preds).flatten()[:500]
    targets = np.concatenate(all_targets).flatten()[:500]
    
    plt.figure(figsize=(15, 6))
    plt.plot(targets, label='True Values (Ground Truth)', color='blue', alpha=0.7)
    plt.plot(preds, label='QLSTM Predictions', color='red', linestyle='--', alpha=0.8)
    plt.title('ECG Signal Prediction vs. Ground Truth', fontsize=16)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Normalized Amplitude', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 预测对比图已保存至: {save_path}")

def plot_phase_space(model, data_loader, device, save_path, delay=5):
    """绘制相空间吸引子重构图"""
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(data_loader))
        data = data.to(device)
        # 使用一批数据进行预测
        preds = model(data).cpu().numpy()
        targets = data.cpu().numpy()

    # 只取第一个样本进行可视化
    pred_series = preds[0].flatten()
    target_series = targets[0].flatten()

    plt.figure(figsize=(8, 8))
    # 绘制真实信号的吸引子
    plt.plot(target_series[:-delay], target_series[delay:], label='Ground Truth Attractor', color='blue', alpha=0.6)
    # 绘制预测信号的吸引子
    plt.plot(pred_series[:-delay], pred_series[delay:], label='Predicted Attractor', color='red', linestyle='--', alpha=0.6)
    
    plt.title(f'Phase Space Reconstruction (delay={delay})', fontsize=16)
    plt.xlabel('x(t)', fontsize=12)
    plt.ylabel(f'x(t+{delay})', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 相空间图已保存至: {save_path}")

def plot_error_distribution(model, data_loader, device, save_path):
    """绘制预测误差的分布直方图"""
    model.eval()
    errors = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            error = (pred - target).cpu().numpy()
            errors.append(error)
            
    errors = np.concatenate(errors).flatten()

    finite_errors = errors[np.isfinite(errors)]
    
    if len(finite_errors) == 0:
        print("⚠️ 警告: 所有预测误差均为非有限值，无法生成误差分布图。")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, density=True, alpha=0.7, color='g', label='Error Distribution')
    
    # 拟合正态分布
    mu, std = norm.fit(errors)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal Distribution')
    
    title = f"Prediction Error Distribution\n(mu = {mu:.4f}, std = {std:.4f})"
    plt.title(title, fontsize=16)
    plt.xlabel('Prediction Error (Pred - True)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 误差分布图已保存至: {save_path}")

def plot_predictions_vs_true(model, data_loader, device, save_path):
    """绘制预测值 vs. 真实值的散点图"""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.3, s=10, label='Predictions')
    # 绘制 y=x 对角线作为参考
    lims = [min(plt.xlim(), plt.ylim()), max(plt.xlim(), plt.ylim())]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Fit (y=x)')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.title('Predictions vs. True Values', fontsize=16)
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 预测-真实散点图已保存至: {save_path}")
