import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import os

def plot_raw_signal_segment(signal, fs, save_path):
    """绘制一小段原始ECG信号"""
    plt.figure(figsize=(15, 5))
    # 只绘制前10秒的数据
    num_samples = int(10 * fs)
    time_axis = np.arange(num_samples) / fs
    plt.plot(time_axis, signal[:num_samples], color='blue')
    plt.title('Raw ECG Signal Segment (First 10 Seconds)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 原始信号片段图已保存至: {save_path}")

def plot_power_spectral_density(signal, fs, save_path):
    """绘制信号的功率谱密度图"""
    f, Pxx = welch(signal, fs, nperseg=1024)
    plt.figure(figsize=(10, 6))
    plt.semilogy(f, Pxx, color='crimson')
    plt.title('Power Spectral Density (PSD)', fontsize=16)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('PSD [V**2/Hz]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, fs / 2) # 只显示到奈奎斯特频率
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 功率谱密度图已保存至: {save_path}")

def plot_signal_histogram(signal, save_path):
    """绘制信号幅值的直方图"""
    plt.figure(figsize=(10, 6))
    plt.hist(signal, bins=100, density=True, color='forestgreen', alpha=0.7)
    plt.title('Signal Amplitude Distribution', fontsize=16)
    plt.xlabel('Amplitude', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 信号幅值分布图已保存至: {save_path}")

def plot_autocorrelation(signal, fs, save_path):
    """绘制信号的自相关图"""
    # 只计算前几秒的自相关以提高效率和清晰度
    segment = signal[:int(5 * fs)]
    corr = np.correlate(segment, segment, mode='full')
    # 只取正延迟部分
    corr = corr[len(corr)//2:]
    
    lag_axis = np.arange(len(corr)) / fs
    
    plt.figure(figsize=(12, 6))
    plt.plot(lag_axis, corr, color='purple')
    plt.title('Autocorrelation of ECG Signal', fontsize=16)
    plt.xlabel('Lag (s)', fontsize=12)
    plt.ylabel('Autocorrelation', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 5) # 只显示前5秒的延迟
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 自相关图已保存至: {save_path}")

def analyze_dataset(signal, fs, results_dir):
    """
    对给定的信号进行一系列可视化分析。
    """
    print("\n" + "="*25 + " 开始数据分析 " + "="*25)
    
    # 创建专门存放数据分析图的子目录
    analysis_plot_dir = os.path.join(results_dir, 'data_analysis_plots')
    if not os.path.exists(analysis_plot_dir):
        os.makedirs(analysis_plot_dir)
        
    plot_raw_signal_segment(signal, fs, os.path.join(analysis_plot_dir, 'raw_signal_segment.png'))
    plot_power_spectral_density(signal, fs, os.path.join(analysis_plot_dir, 'power_spectral_density.png'))
    plot_signal_histogram(signal, os.path.join(analysis_plot_dir, 'signal_histogram.png'))
    plot_autocorrelation(signal, fs, os.path.join(analysis_plot_dir, 'autocorrelation.png'))
    
    print("="*25 + " 数据分析完成 " + "="*25 + "\n")