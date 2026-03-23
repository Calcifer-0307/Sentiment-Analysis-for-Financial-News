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
    
    # ==========================================
    # 2. Global Training Settings
    # ==========================================
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LABEL_MAP = {'positive': 0, 'negative': 1, 'neutral': 2}
    TARGET_NAMES = ['positive', 'negative', 'neutral']
    # Handling imbalanced classes (Neutral: 2879, Positive: 1363, Negative: 604)
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
    BERT_BATCH_SIZE = 16
    BERT_LR = 2e-5
    BERT_EPOCHS = 4
    
    # ==========================================
    # 5. Preprocessing Settings
    # ==========================================
    TRAIN_RATIO = 0.75
