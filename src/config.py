import torch
import os

class Config:
    # ==========================================
    # 1. Path Configuration
    # ==========================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'raw')

    TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data.csv')
    TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.csv')
    RAW_DATA_PATH = os.path.join(DATA_DIR, 'all-data.csv')
    PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')

    # 输出目录（保存模型、结果、预测表）
    OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'outputs')

    # 当前实验名字（每次实验都改这个，方便保存结果）
    EXP_NAME = 'baseline'

    # ==========================================
    # 2. Global Training Settings
    # ==========================================
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LABEL_MAP = {'positive': 0, 'negative': 1, 'neutral': 2}
    TARGET_NAMES = ['positive', 'negative', 'neutral']

    # 类别权重（按你当前标签顺序：positive, negative, neutral）
    # negative 样本最少，所以权重更大
    CLASS_WEIGHTS = [1.0, 2.5, 1.0]

    # ==========================================
    # 3. TextCNN Hyperparameters
    # ==========================================
    CNN_MAX_LEN = 32
    CNN_BATCH_SIZE = 64
    CNN_LR = 1e-3
    CNN_EPOCHS = 10
    CNN_EMBEDDING_DIM = 100
    CNN_NUM_FILTERS = 100
    CNN_FILTER_SIZES = [3, 4, 5]
    CNN_DROPOUT = 0.5

    # ==========================================
    # 4. DistilBERT Hyperparameters
    # ==========================================
    BERT_MODEL_NAME = 'distilbert-base-uncased'
    BERT_MAX_LEN = 64

    # baseline 默认设置
    BERT_BATCH_SIZE = 16
    BERT_LR = 2e-5

    # 建议增加 epoch，方便观察收敛曲线
    BERT_EPOCHS = 8

    # 选择损失函数（后续做 loss ablation 时改这里）
    LOSS_NAME = 'cross_entropy'
    # 可选值：
    # 'cross_entropy'
    # 'focal'

    # ==========================================
    # 5. Attention LSTM Hyperparameters
    # ==========================================
    LSTM_MAX_LEN = 32
    LSTM_BATCH_SIZE = 32
    LSTM_LR = 0.001
    LSTM_EPOCHS = 10
    LSTM_EMBEDDING_DIM = 128
    LSTM_HIDDEN_DIM = 128
    LSTM_DROPOUT = 0.5

    # ==========================================
    # 6. Data Split / Preprocessing Settings
    # ==========================================
    # 原始 train_data.csv 再划分一部分作为 validation
    VAL_SIZE = 0.1

    # preprocess.py 里 train/test 划分比例
    TRAIN_RATIO = 0.75
