import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from config import Config

# ==========================================
# 1. Device & Random Seed Configuration
# ==========================================
print(f"Using device: {Config.DEVICE}")

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 让训练尽量可复现
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

set_seed(Config.SEED)

# Create output dirs
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(Config.OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(Config.OUTPUT_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(Config.OUTPUT_DIR, "predictions"), exist_ok=True)

# ==========================================
# 2. Load and Preprocess Data
# ==========================================
if not os.path.exists(Config.TRAIN_DATA_PATH) or not os.path.exists(Config.TEST_DATA_PATH):
    raise FileNotFoundError("Preprocessed data not found. Please run src/preprocess.py first.")

full_train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
test_df = pd.read_csv(Config.TEST_DATA_PATH)

# Label mapping
full_train_df["label"] = full_train_df["sentiment"].map(Config.LABEL_MAP)
test_df["label"] = test_df["sentiment"].map(Config.LABEL_MAP)

# Handle missing values
full_train_df["processed_text"] = full_train_df["processed_text"].fillna("").astype(str)
test_df["processed_text"] = test_df["processed_text"].fillna("").astype(str)

# Train / Validation split
train_df, val_df = train_test_split(
    full_train_df,
    test_size=Config.VAL_SIZE,
    stratify=full_train_df["label"],
    random_state=Config.SEED
)

X_train = train_df["processed_text"].values
y_train = train_df["label"].values

X_val = val_df["processed_text"].values
y_val = val_df["label"].values

X_test = test_df["processed_text"].values
y_test = test_df["label"].values

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
print("Class distribution in training set:", np.bincount(y_train))

# ==========================================
# 3. Build Dataset & DataLoader
# ==========================================
class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

tokenizer = DistilBertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
bert_model = DistilBertForSequenceClassification.from_pretrained(
    Config.BERT_MODEL_NAME,
    num_labels=len(Config.TARGET_NAMES)
)
bert_model.to(Config.DEVICE)

train_dataset = BertDataset(X_train, y_train, tokenizer, max_len=Config.BERT_MAX_LEN)
val_dataset = BertDataset(X_val, y_val, tokenizer, max_len=Config.BERT_MAX_LEN)
test_dataset = BertDataset(X_test, y_test, tokenizer, max_len=Config.BERT_MAX_LEN)

# 固定 DataLoader 随机性（仅 train_loader 用 shuffle）
g = torch.Generator()
g.manual_seed(Config.SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BERT_BATCH_SIZE,
    shuffle=True,
    generator=g
)
val_loader = DataLoader(
    val_dataset,
    batch_size=Config.BERT_BATCH_SIZE,
    shuffle=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=Config.BERT_BATCH_SIZE,
    shuffle=False
)

# ==========================================
# 4. Optimizer and Loss Function
# ==========================================
optimizer = AdamW(bert_model.parameters(), lr=Config.BERT_LR)

class_weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float).to(Config.DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# ==========================================
# 5. Training & Evaluation Functions
# ==========================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(Config.DEVICE)
        attention_mask = batch["attention_mask"].to(Config.DEVICE)
        labels = batch["labels"].to(Config.DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(logits, dim=1)
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
        for batch in loader:
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            labels = batch["labels"].to(Config.DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item()

            _, preds = torch.max(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc


def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            labels = batch["labels"].to(Config.DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

# ==========================================
# 6. Training Loop with Early Stopping
# ==========================================
if __name__ == "__main__":
    print("\nStarting training...")

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    test_losses, test_accs = [], []

    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    best_model_path = os.path.join(Config.OUTPUT_DIR, "models", f"best_{Config.EXP_NAME}.pt")

    for epoch in range(Config.BERT_EPOCHS):
        train_loss, train_acc = train_epoch(bert_model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(bert_model, val_loader, criterion)
        test_loss, test_acc = evaluate(bert_model, test_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(
            f"Epoch {epoch+1}/{Config.BERT_EPOCHS}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "
            f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}"
        )

        # 如果 val_acc 明显提升，则保存模型
        if val_acc > best_val_acc + Config.EARLY_STOPPING_MIN_DELTA:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(bert_model.state_dict(), best_model_path)
            print(f"  --> New best model saved at epoch {best_epoch} (Val Acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement for {epochs_no_improve} epoch(s)")

        # Early stopping
        if epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    print(f"\nBest model saved to: {best_model_path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # ==========================================
    # 7. Save Metrics for Plotting
    # ==========================================
    results = {
        "exp_name": Config.EXP_NAME,
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs,
        "test_loss": test_losses,
        "test_acc": test_accs,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "lr": Config.BERT_LR,
        "batch_size": Config.BERT_BATCH_SIZE,
        "epochs": Config.BERT_EPOCHS,
        "seed": Config.SEED
    }

    result_path = os.path.join(Config.OUTPUT_DIR, "results", f"results_{Config.EXP_NAME}.pkl")
    with open(result_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Training curves saved to: {result_path}")

    # ==========================================
    # 8. Load Best Model and Evaluate on Test
    # ==========================================
    bert_model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

    print("\nGenerating final classification report using BEST model...")
    y_pred, y_true = get_predictions(bert_model, test_loader)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=Config.TARGET_NAMES))

    # Save first 100 predictions
    pred_df = test_df.copy().reset_index(drop=True)
    pred_df["true_label"] = [Config.TARGET_NAMES[i] for i in y_true]
    pred_df["pred_label"] = [Config.TARGET_NAMES[i] for i in y_pred]
    pred_df["correct"] = pred_df["true_label"] == pred_df["pred_label"]

    pred_100_path = os.path.join(Config.OUTPUT_DIR, "predictions", f"pred_100_{Config.EXP_NAME}.csv")
    pred_df.head(100).to_csv(pred_100_path, index=False)
    print(f"Top-100 prediction file saved to: {pred_100_path}")

    print("Training complete.")
