import os
import torch
import logging
from datetime import datetime
import config
from src.data_loader import create_data_loaders
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.utils.visualization import (
    plot_training_curves, plot_predictions, 
    plot_phase_space, plot_error_distribution, plot_predictions_vs_true
)
import wfdb
from src.utils.data_analysis import analyze_dataset
# import numpy as np

def setup_logging(results_dir):
    log_file = os.path.join(results_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def main():
    # 创建结果目录
    timestamp = datetime.now().strftime("%m%d_%H%M")
    results_dir = os.path.join(config.RESULTS_DIR, f"{config.MODEL_TYPE}_{config.DATASET_CONFIG['record_name']}_{timestamp}")
    model_dir = config.MODEL_DIR
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    
    logger = setup_logging(results_dir)
    logger.info(f"开始运行 - 模型类型: {config.MODEL_TYPE}")

     # --- 新增：数据分析部分 ---
    raw_data_path = os.path.join('data/raw', config.DATASET_CONFIG['record_name'])
    if os.path.exists(raw_data_path + '.dat'):
        logger.info(f"正在加载原始数据进行分析: {raw_data_path}")
        record = wfdb.rdrecord(raw_data_path)
        raw_signal = record.p_signal[:, 0]
        fs = record.fs
        analyze_dataset(raw_signal, fs, results_dir)
    else:
        logger.warning(f"原始数据文件未找到于 {raw_data_path}，跳过数据分析。请先运行data_processing.py下载数据。")

    # 加载数据
    train_data_path = config.TRAIN_DATA_PATH
    val_data_path = config.VAL_DATA_PATH
    if not os.path.exists(train_data_path):
        logger.error(f"数据文件 '{train_data_path}' 不存在，请先运行 data_processing.py")
        return

    logger.info(f"正在从以下路径加载数据:\n  训练集: {train_data_path}\n  验证集: {val_data_path}")
    train_loader, val_loader = create_data_loaders(train_data_path, val_data_path, config.BATCH_SIZE, None)

    # 创建模型
    model_config_dict = {'qlstm_config': config.QLSTM_CONFIG}
    model = create_model(config.MODEL_TYPE, model_config_dict)
    
    # 训练配置
    trainer_config = {
        'device': config.DEVICE,
        'epochs': config.EPOCHS,
        'training_config': config.TRAINING_CONFIG,
        'model_dir': model_dir,
        'model_type': config.MODEL_TYPE
    }

    # 训练模型
    trainer = Trainer(model, trainer_config)
    train_losses, val_losses = trainer.train(train_loader, val_loader)

    # 绘制训练曲线
    logger.info("📈 正在生成结果图表...")
    plot_training_curves(train_losses, val_losses, os.path.join(results_dir, 'training_curves.png'))
    
    # 加载最佳模型进行后续分析
    best_model_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_TYPE}_best.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        device = torch.device(config.DEVICE)
        
        # 调用所有绘图函数
        plot_predictions(model, val_loader, device, os.path.join(results_dir, 'predictions.png'))
        plot_phase_space(model, val_loader, device, os.path.join(results_dir, 'phase_space.png'))
        plot_error_distribution(model, val_loader, device, os.path.join(results_dir, 'error_distribution.png'))
        plot_predictions_vs_true(model, val_loader, device, os.path.join(results_dir, 'predictions_vs_true.png'))
    
    logger.info(f"✅ 训练完成，所有结果保存在: {results_dir}")

if __name__ == "__main__":
    main()