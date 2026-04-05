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

    # ==========================================
    # 2. Global Training Settings
    # ==========================================
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LABEL_MAP = {
        'positive': 0,
        'negative': 1,
        'neutral': 2
    }

    TARGET_NAMES = ['positive', 'negative', 'neutral']

    # 顺序必须和 LABEL_MAP 对应
    # positive -> 1.0
    # negative -> 2.5
    # neutral  -> 1.0
    CLASS_WEIGHTS = [1.0, 2.5, 1.0]

    # ==========================================
    # 3. Preprocessing / Data Split Settings
    # ==========================================
    TRAIN_RATIO = 0.75
    VAL_SIZE = 0.1

    # ==========================================
    # 4. Experiment Name
    # ==========================================
    # ============================
    # 【实验时要改】
    # 每次实验都改这个名字，避免结果文件被覆盖
    # 例如：
    # 'cnn_baseline'
    # 'cnn_lr_0.1'
    # 'cnn_lr_0.01'
    # 'cnn_lr_0.001'
    # 'cnn_lr_0.0001'
    # 'cnn_bs_8'
    # 'cnn_bs_16'
    # 'cnn_bs_32'
    # 'cnn_bs_64'
    # 'cnn_bs_128'
    # 'cnn_focal'
    # ============================
    EXP_NAME = 'cnn_bs_32'

    # ==========================================
    # 5. TextCNN Hyperparameters
    # ==========================================
    CNN_MAX_LEN = 32

    # ============================
    # 【实验时要改：Batch Size 对比实验改这里】
    # baseline 可以先用 64
    # 可测试：
    # 8 / 16 / 32 / 64 / 128
    # ============================
    CNN_BATCH_SIZE = 32

    # ============================
    # 【实验时要改：Learning Rate 对比实验改这里】
    # baseline 建议先用 1e-3 (= 0.001)
    # 可测试：
    # 0.1 / 0.01 / 0.001 / 0.0001
    # ============================
    CNN_LR = 0.01

    # ============================
    # 【建议】
    # 如果你要做完整曲线分析，epoch 不建议太少
    # 推荐 20 或 30
    # 如果时间紧可以先 15~20
    # ============================
    CNN_EPOCHS = 100

    CNN_EMBEDDING_DIM = 100
    CNN_NUM_FILTERS = 100
    CNN_FILTER_SIZES = [3, 4, 5]
    CNN_DROPOUT = 0.5

    # ============================
    # 【实验时要改：Loss Function 对比实验改这里】
    # 可选：
    # 'cross_entropy'
    # 'focal'
    # baseline 先用 cross_entropy
    # ============================
    CNN_LOSS_NAME = 'cross_entropy'

    # ==========================================
    # 6. CNN Early Stopping Settings（可选）
    # ==========================================
    # 如果你后面想让 CNN 自动停，可以在 train_CNN.py 里使用这两个参数
    CNN_EARLY_STOPPING_PATIENCE = 3
    CNN_EARLY_STOPPING_MIN_DELTA = 0.001

    # ==========================================
    # 7. DistilBERT Hyperparameters
    # ==========================================
    BERT_MODEL_NAME = 'distilbert-base-uncased'
    BERT_MAX_LEN = 64
    BERT_BATCH_SIZE = 16
    BERT_LR = 2e-5
    BERT_EPOCHS = 100

    # 可选：'cross_entropy' 或 'focal'
    LOSS_NAME = 'cross_entropy'

    # ==========================================
    # 8. Transformer Early Stopping Settings
    # ==========================================
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_DELTA = 0.001

    # ==========================================
    # 9. Attention LSTM Hyperparameters
    # ==========================================
    LSTM_MAX_LEN = 32
    LSTM_BATCH_SIZE = 32
    LSTM_LR = 0.001
    LSTM_EPOCHS = 10
    LSTM_EMBEDDING_DIM = 128
    LSTM_HIDDEN_DIM = 128
    LSTM_DROPOUT = 0.5

    # ==========================================
    # 10. Optional Saving Flags
    # ==========================================
    SAVE_RESULTS = True
    SAVE_PREDICTIONS = True
    SAVE_BEST_MODEL = True
