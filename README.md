# CAQLSTM - 混沌增强量子LSTM模型

基于混沌特征增强的量子LSTM模型，用于时间序列预测任务。

## 项目结构

```
CAQLSTM/
├── config.py              # 配置文件
├── main.py                # 主程序入口
├── data_processing.py     # 数据处理脚本
├── requirements.txt       # 依赖包列表
├── src/
│   ├── models/           # 模型定义
│   ├── training/         # 训练相关
│   ├── utils/            # 工具函数
│   └── data_loader.py    # 数据加载器
├── data/                 # 数据目录
├── results/              # 结果输出目录
└── models/               # 模型保存目录
```

## 安装

1. 克隆项目：
```bash
git clone <repository-url>
cd CAQLSTM
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 数据处理：
```bash
python data_processing.py
```

2. 训练模型：
```bash
python main.py
```

## 配置

在 `config.py` 中可以调整以下参数：
- `QUICK_TEST_MODE`: 快速测试模式开关
- `DATASET_CONFIG`: 数据集配置
- `QLSTM_CONFIG`: 量子LSTM模型配置
- `TRAINING_CONFIG`: 训练配置

## 特性

- 混沌特征提取（样本熵、DFA、相关维数）
- 量子LSTM模型
- 自动数据预处理
- 训练过程可视化
- 结果分析和图表生成

## 依赖

- PyTorch
- TorchQuantum
- NumPy
- Matplotlib
- SciPy
- WFDB
- Nolds 