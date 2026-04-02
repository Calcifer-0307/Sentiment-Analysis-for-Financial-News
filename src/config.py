import os
import torch

class Config:
    # ==========================================
    # 1. Path Configuration
    # ==========================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

    TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data.csv')
    TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.csv')
    RAW_DATA_PATH = os.path.join(DATA_DIR, 'all-data.csv')
    PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')

    # 当前实验名字（每次实验都改这个，避免结果文件被覆盖）
    # baseline / lr_0.001 / batch_32 / focal 等
    EXP_NAME = 'baseline'

    # ==========================================
    # 2. Global Training Settings
    # ==========================================
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 统一标签映射（全项目都尽量用这一套）
    LABEL_MAP = {
        'positive': 0,
        'negative': 1,
        'neutral': 2
    }

    TARGET_NAMES = ['positive', 'negative', 'neutral']

    # 类别权重（顺序必须和 LABEL_MAP 对应）
    # positive -> 1.0
    # negative -> 2.5   (负类最少，权重大一些)
    # neutral  -> 1.0
    CLASS_WEIGHTS = [1.0, 2.5, 1.0]

    # ==========================================
    # 3. Preprocessing / Data Split Settings
    # ==========================================
    # preprocess.py 里 train/test 的比例
    TRAIN_RATIO = 0.75

    # train_data.csv 再划出 validation 的比例
    VAL_SIZE = 0.1

    # ==========================================
    # 4. TextCNN Hyperparameters
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
    # 5. DistilBERT Hyperparameters
    # ==========================================
    BERT_MODEL_NAME = 'distilbert-base-uncased'
    BERT_MAX_LEN = 64

    # baseline 默认设置
    BERT_BATCH_SIZE = 16
    BERT_LR = 2e-5

    # 最大训练轮数（配合 early stopping，一般不会真的跑满）
    BERT_EPOCHS = 100

    # 损失函数选择（后面做 loss ablation 时改这里）
    # 可选：
    # 'cross_entropy'
    # 'focal'
    LOSS_NAME = 'cross_entropy'

    # ==========================================
    # 6. Early Stopping Settings
    # ==========================================
    # 如果连续若干轮没有明显提升，就自动停止
    EARLY_STOPPING_PATIENCE = 3

    # 至少提升这么多才算“有效提升”
    EARLY_STOPPING_MIN_DELTA = 0.001

    # ==========================================
    # 7. Attention LSTM Hyperparameters
    # ==========================================
    LSTM_MAX_LEN = 32
    LSTM_BATCH_SIZE = 32
    LSTM_LR = 0.001
    LSTM_EPOCHS = 10
    LSTM_EMBEDDING_DIM = 128
    LSTM_HIDDEN_DIM = 128
    LSTM_DROPOUT = 0.5

    # ==========================================
    # 8. Optional Saving Flags
    # ==========================================
    SAVE_RESULTS = True
    SAVE_PREDICTIONS = True
    SAVE_BEST_MODEL = True
``
