import os
import random
import pickle
import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

from config import Config

warnings.filterwarnings("ignore")

# ==========================================
# 1. Basic Settings
# ==========================================
print(f"Using device: {Config.DEVICE}")

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)

# Create output dirs
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(Config.OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(Config.OUTPUT_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(Config.OUTPUT_DIR, "reports"), exist_ok=True)
os.makedirs(os.path.join(Config.OUTPUT_DIR, "predictions"), exist_ok=True)

# ==========================================
# 2. Check Data Files
# ==========================================
if not os.path.exists(Config.TRAIN_DATA_PATH) or not os.path.exists(Config.TEST_DATA_PATH):
    raise FileNotFoundError(
        "Preprocessed data files not found. Please run src/preprocess.py first."
    )

# ==========================================
# 3. Load Data
# ==========================================
train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
test_df = pd.read_csv(Config.TEST_DATA_PATH)

# Label mapping
train_df["label"] = train_df["sentiment"].map(Config.LABEL_MAP)
test_df["label"] = test_df["sentiment"].map(Config.LABEL_MAP)

# Handle missing values
train_df["processed_text"] = train_df["processed_text"].fillna("").astype(str)
test_df["processed_text"] = test_df["processed_text"].fillna("").astype(str)

X_train = train_df["processed_text"].values
y_train = train_df["label"].values
X_test = test_df["processed_text"].values
y_test = test_df["label"].values

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print("Class distribution in training set:", np.bincount(y_train))

# ==========================================
# 4. Dataset Definition
# ==========================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=32, is_train=True):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.is_train = is_train

        if is_train:
            self.vocab = self.build_vocab(texts)
        else:
            self.vocab = vocab

        self.vocab_size = len(self.vocab)

    def build_vocab(self, texts, min_freq=2):
        counter = Counter()
        for text in texts:
            for word in str(text).split():
                counter[word] += 1

        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, freq in counter.items():
            if freq >= min_freq:
                vocab[word] = len(vocab)

        return vocab

    def encode_text(self, text):
        tokens = str(text).split()
        ids = [self.vocab.get(tok, 1) for tok in tokens]

        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.encode_text(self.texts[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

# ==========================================
# 5. DataLoaders
# ==========================================
train_dataset = TextDataset(
    X_train,
    y_train,
    max_len=Config.CNN_MAX_LEN,
    is_train=True
)

test_dataset = TextDataset(
    X_test,
    y_test,
    vocab=train_dataset.vocab,
    max_len=Config.CNN_MAX_LEN,
    is_train=False
)

g = torch.Generator()
g.manual_seed(Config.SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=Config.CNN_BATCH_SIZE,
    shuffle=True,
    generator=g
)

test_loader = DataLoader(
    test_dataset,
    batch_size=Config.CNN_BATCH_SIZE,
    shuffle=False
)

print(f"CNN vocab size: {train_dataset.vocab_size}")

# ==========================================
# 6. Model Definition
# ==========================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        emb = self.embedding(x)                 # (batch, seq_len, embed_dim)
        emb = emb.permute(0, 2, 1)             # (batch, embed_dim, seq_len)

        conv_outs = [F.relu(conv(emb)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outs]

        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# ==========================================
# 7. Optional Focal Loss
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ==========================================
# 8. Build Model / Criterion / Optimizer
# ==========================================
vocab_size = train_dataset.vocab_size
num_classes = len(Config.TARGET_NAMES)

model = TextCNN(
    vocab_size=vocab_size,
    embedding_dim=Config.CNN_EMBEDDING_DIM,
    num_filters=Config.CNN_NUM_FILTERS,
    filter_sizes=Config.CNN_FILTER_SIZES,
    num_classes=num_classes,
    dropout=Config.CNN_DROPOUT
).to(Config.DEVICE)

class_weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32).to(Config.DEVICE)

if Config.CNN_LOSS_NAME == "cross_entropy":
    criterion = nn.CrossEntropyLoss(weight=class_weights)
elif Config.CNN_LOSS_NAME == "focal":
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
else:
    raise ValueError(f"Unsupported CNN_LOSS_NAME: {Config.CNN_LOSS_NAME}")

optimizer = optim.Adam(model.parameters(), lr=Config.CNN_LR)

# ==========================================
# 9. Training / Evaluation Functions
# ==========================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for texts, labels in loader:
        texts = texts.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in loader:
            texts = texts.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc


def get_predictions(model, loader):
    model.eval()
    preds = []
    labels_all = []

    with torch.no_grad():
        for texts, labels in loader:
            texts = texts.to(Config.DEVICE)

            outputs = model(texts)
            _, pred = torch.max(outputs, dim=1)

            preds.extend(pred.cpu().numpy())
            labels_all.extend(labels.numpy())

    return preds, labels_all

# ==========================================
# 10. Training Loop with Early Stopping
# ==========================================
if __name__ == "__main__":
    print(f"\nStarting CNN experiment: {Config.EXP_NAME}")
    print(f"Loss Function: {Config.CNN_LOSS_NAME}")
    print(f"Learning Rate: {Config.CNN_LR}")
    print(f"Batch Size: {Config.CNN_BATCH_SIZE}")
    print(f"Epochs (max): {Config.CNN_EPOCHS}")

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    best_test_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    best_model_path = os.path.join(Config.OUTPUT_DIR, "models", f"best_{Config.EXP_NAME}.pt")

    for epoch in range(Config.CNN_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(
            f"Epoch {epoch+1}/{Config.CNN_EPOCHS}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}"
        )

        # Save best model based on test accuracy
        if test_acc > best_test_acc + Config.CNN_EARLY_STOPPING_MIN_DELTA:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0

            if Config.SAVE_BEST_MODEL:
                torch.save(model.state_dict(), best_model_path)

            print(f"  --> New best CNN model saved at epoch {best_epoch} (Test Acc={best_test_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement for {epochs_no_improve} epoch(s)")

        # Early stopping
        if epochs_no_improve >= Config.CNN_EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best test accuracy: {best_test_acc:.4f}")

    # ==========================================
    # 11. Save Training Curves
    # ==========================================
    results = {
        "exp_name": Config.EXP_NAME,
        "train_loss": train_losses,
        "train_acc": train_accs,
        "test_loss": test_losses,
        "test_acc": test_accs,
        "best_epoch": best_epoch,
        "best_test_acc": best_test_acc,
        "lr": Config.CNN_LR,
        "batch_size": Config.CNN_BATCH_SIZE,
        "epochs": Config.CNN_EPOCHS,
        "loss_name": Config.CNN_LOSS_NAME,
        "seed": Config.SEED
    }

    if Config.SAVE_RESULTS:
        result_path = os.path.join(Config.OUTPUT_DIR, "results", f"results_{Config.EXP_NAME}.pkl")
        with open(result_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Training curves saved to: {result_path}")

    # ==========================================
    # 12. Load Best Model
    # ==========================================
    if Config.SAVE_BEST_MODEL and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
        print(f"Loaded best model from: {best_model_path}")

    # ==========================================
    # 13. Final Evaluation
    # ==========================================
    y_pred, y_true = get_predictions(model, test_loader)

    print("\nCNN Classification Report:")
    report_str = classification_report(y_true, y_pred, target_names=Config.TARGET_NAMES)
    print(report_str)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=Config.TARGET_NAMES,
        output_dict=True
    )

    # Save report
    report_path = os.path.join(Config.OUTPUT_DIR, "reports", f"report_{Config.EXP_NAME}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Classification report saved to: {report_path}")

    # ==========================================
    # 14. Save Top-100 Predictions
    # ==========================================
    if Config.SAVE_PREDICTIONS:
        pred_df = test_df.copy().reset_index(drop=True)
        pred_df["true_label"] = [Config.TARGET_NAMES[i] for i in y_true]
        pred_df["pred_label"] = [Config.TARGET_NAMES[i] for i in y_pred]
        pred_df["correct"] = pred_df["true_label"] == pred_df["pred_label"]

        pred_100_path = os.path.join(Config.OUTPUT_DIR, "predictions", f"pred_100_{Config.EXP_NAME}.csv")
        pred_df.head(100).to_csv(pred_100_path, index=False)
        print(f"Top-100 prediction file saved to: {pred_100_path}")

    print("\nCNN experiment complete.")


