import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import classification_report
from config import Config
import time

# ==========================================
# 1. Device & Data Loading Configuration
# ==========================================
print(f"Using device: {Config.DEVICE}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(Config.SEED)

if not os.path.exists(Config.TRAIN_DATA_PATH) or not os.path.exists(Config.TEST_DATA_PATH):
    raise FileNotFoundError("Preprocessed data not found. Please run src/preprocess.py first.")

# ==========================================
# 2. Helper Functions for Vocabulary & Encoding
# ==========================================
def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(str(text).split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode_texts(texts, vocab, max_len):
    encoded = []
    for text in texts:
        tokens = str(text).split()
        ids = [vocab.get(token, 1) for token in tokens]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)

def get_attention_lstm_dataloaders(train_path, test_path, batch_size, max_len):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Create label column
    train_df['label'] = train_df['sentiment'].map(Config.LABEL_MAP)
    test_df['label'] = test_df['sentiment'].map(Config.LABEL_MAP)

    train_df['processed_text'] = train_df['processed_text'].fillna('').astype(str)
    test_df['processed_text'] = test_df['processed_text'].fillna('').astype(str)

    X_train = train_df['processed_text'].values
    y_train = train_df['label'].values
    X_test = test_df['processed_text'].values
    y_test = test_df['label'].values

    vocab = build_vocab(X_train)
    X_train_enc = encode_texts(X_train, vocab, max_len)
    X_test_enc = encode_texts(X_test, vocab, max_len)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train_enc, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test_enc, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vocab

# ==========================================
# 3. Define Attention LSTM model
# ==========================================
class AttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5):
        super(AttentionLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)  # bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)     # (batch, seq_len, hidden_dim*2)
        # Attention
        attn_weights = torch.tanh(self.attention(lstm_out))  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted = lstm_out * attn_weights
        context = torch.sum(weighted, dim=1)   # (batch, hidden_dim*2)
        context = self.dropout(context)
        out = self.fc(context)
        return out

# ==========================================
# 4. Training and evaluation functions
# ==========================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

if __name__ == "__main__":
    # ==========================================
    # 5. Load Data & Initialize Model
    # ==========================================
    print("Loading data...")
    train_loader, test_loader, vocab = get_attention_lstm_dataloaders(
        Config.TRAIN_DATA_PATH,
        Config.TEST_DATA_PATH,
        batch_size=Config.LSTM_BATCH_SIZE,
        max_len=Config.LSTM_MAX_LEN
    )
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    model = AttentionLSTM(
        vocab_size, 
        Config.LSTM_EMBEDDING_DIM, 
        Config.LSTM_HIDDEN_DIM, 
        len(Config.TARGET_NAMES),
        dropout=Config.LSTM_DROPOUT
    ).to(Config.DEVICE)
    
    # Apply class weights to loss function
    class_weights = torch.tensor(Config.CLASS_WEIGHTS).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=Config.LSTM_LR)

    # ==========================================
    # 6. Training Loop
    # ==========================================
    print("\n--- Training Attention LSTM ---")
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    start_time = time.time()

    for epoch in range(Config.LSTM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{Config.LSTM_EPOCHS}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds")

    # ==========================================
    # 7. Final Evaluation
    # ==========================================
    y_pred, y_true = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(Config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    print("\nClassification Report (Attention LSTM):")
    print(classification_report(y_true, y_pred, target_names=Config.TARGET_NAMES))