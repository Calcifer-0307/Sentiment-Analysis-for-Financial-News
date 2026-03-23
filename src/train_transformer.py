import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report
from config import Config

# ==========================================
# 1. Device & Random Seed Configuration
# ==========================================
print(f"Using device: {Config.DEVICE}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(Config.SEED)

# ==========================================
# 2. Load and Preprocess Data
# ==========================================
if not os.path.exists(Config.TRAIN_DATA_PATH) or not os.path.exists(Config.TEST_DATA_PATH):
    raise FileNotFoundError("Preprocessed data not found. Please run src/preprocess.py first.")

train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
test_df = pd.read_csv(Config.TEST_DATA_PATH)

# Label mapping
train_df['label'] = train_df['sentiment'].map(Config.LABEL_MAP)
test_df['label'] = test_df['sentiment'].map(Config.LABEL_MAP)

# Handle missing values and ensure string type
train_df['processed_text'] = train_df['processed_text'].fillna('').astype(str)
test_df['processed_text'] = test_df['processed_text'].fillna('').astype(str)

X_train = train_df['processed_text'].values
y_train = train_df['label'].values
X_test = test_df['processed_text'].values
y_test = test_df['label'].values

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
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
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
bert_model = DistilBertForSequenceClassification.from_pretrained(Config.BERT_MODEL_NAME, num_labels=len(Config.TARGET_NAMES))
bert_model.to(Config.DEVICE)

# Create Datasets and DataLoaders
train_dataset = BertDataset(X_train, y_train, tokenizer, max_len=Config.BERT_MAX_LEN)
test_dataset = BertDataset(X_test, y_test, tokenizer, max_len=Config.BERT_MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=Config.BERT_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=Config.BERT_BATCH_SIZE, shuffle=False)

# ==========================================
# 4. Optimizer and Loss Function
# ==========================================
optimizer = AdamW(bert_model.parameters(), lr=Config.BERT_LR)
# Use class weights to handle imbalanced data
class_weights = torch.tensor(Config.CLASS_WEIGHTS).to(Config.DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# ==========================================
# 5. Training & Evaluation Functions
# ==========================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(Config.DEVICE)
        attention_mask = batch['attention_mask'].to(Config.DEVICE)
        labels = batch['labels'].to(Config.DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, pred = torch.max(logits, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
    return total_loss / len(loader), correct / total

# ==========================================
# 6. Training Loop
# ==========================================
if __name__ == "__main__":
    print("\nStarting training...")
    for epoch in range(Config.BERT_EPOCHS):
        train_loss = train_epoch(bert_model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(bert_model, test_loader, criterion)
        print(f"Epoch {epoch+1}/{Config.BERT_EPOCHS}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

    # ==========================================
    # 7. Final Evaluation & Classification Report
    # ==========================================
    def get_predictions(model, loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                labels = batch['labels'].to(Config.DEVICE)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return all_preds, all_labels

    print("\nGenerating final classification report...")
    y_pred, y_true = get_predictions(bert_model, test_loader)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=Config.TARGET_NAMES))
    print("Training complete.")
