

import os
import sys
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
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
from config import Config
warnings.filterwarnings('ignore')

# 1. Download NLTK data if needed
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# 2. Check data files

if not os.path.exists(Config.TRAIN_DATA_PATH) or not os.path.exists(Config.TEST_DATA_PATH):
    print("Error: Preprocessed data files not found.")
    print("Please run src/preprocess.py first to generate training/test data.")
    sys.exit(1)

# 3. Global settings
print(f"Using device: {Config.DEVICE}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(Config.SEED)

# 4. Load data

train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
test_df = pd.read_csv(Config.TEST_DATA_PATH)

# Label mapping
train_df['label'] = train_df['sentiment'].map(Config.LABEL_MAP)
test_df['label'] = test_df['sentiment'].map(Config.LABEL_MAP)

# Ensure processed_text column is string (handle missing values)
train_df['processed_text'] = train_df['processed_text'].fillna('').astype(str)
test_df['processed_text'] = test_df['processed_text'].fillna('').astype(str)

X_train = train_df['processed_text'].values
y_train = train_df['label'].values
X_test = test_df['processed_text'].values
y_test = test_df['label'].values

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print("Class distribution in training set:", np.bincount(y_train))

# Part I: TextCNN Model

# 5. CNN Dataset definition
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
        vocab = {'<PAD>': 0, '<UNK>': 1}
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

# 6. Create CNN DataLoaders
train_dataset_cnn = TextDataset(X_train, y_train, max_len=Config.CNN_MAX_LEN, is_train=True)
test_dataset_cnn = TextDataset(X_test, y_test, vocab=train_dataset_cnn.vocab, max_len=Config.CNN_MAX_LEN, is_train=False)

train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=Config.CNN_BATCH_SIZE, shuffle=True)
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=Config.CNN_BATCH_SIZE, shuffle=False)
print(f"CNN vocab size: {train_dataset_cnn.vocab_size}")

# 7. Define TextCNN model
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
        emb = self.embedding(x)                     # (batch, seq_len, embed_dim)
        emb = emb.permute(0, 2, 1)                  # (batch, embed_dim, seq_len)
        conv_outs = [F.relu(conv(emb)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outs]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# 8. Model parameters and optimizer
vocab_size = train_dataset_cnn.vocab_size
num_classes = len(Config.TARGET_NAMES)

model_cnn = TextCNN(
    vocab_size, 
    Config.CNN_EMBEDDING_DIM, 
    Config.CNN_NUM_FILTERS, 
    Config.CNN_FILTER_SIZES, 
    num_classes,
    Config.CNN_DROPOUT
).to(Config.DEVICE)

criterion_cnn = nn.CrossEntropyLoss(weight=torch.tensor(Config.CLASS_WEIGHTS).to(Config.DEVICE))
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=Config.CNN_LR)

# 9. Training functions
def train_cnn_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for texts, labels in loader:
        texts, labels = texts.to(Config.DEVICE), labels.to(Config.DEVICE)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_cnn(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return total_loss / len(loader), correct / total

# 10. Train CNN
print("\nTraining TextCNN...")
for epoch in range(Config.CNN_EPOCHS):
    train_loss = train_cnn_epoch(model_cnn, train_loader_cnn, criterion_cnn, optimizer_cnn)
    test_loss, test_acc = evaluate_cnn(model_cnn, test_loader_cnn, criterion_cnn)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

# 11. CNN classification report
def get_cnn_predictions(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for texts, lbls in loader:
            texts = texts.to(Config.DEVICE)
            outputs = model(texts)
            _, pred = torch.max(outputs, 1)
            preds.extend(pred.cpu().numpy())
            labels.extend(lbls.numpy())
    return preds, labels

y_pred_cnn, y_true_cnn = get_cnn_predictions(model_cnn, test_loader_cnn)
print("\nCNN Classification Report:")
print(classification_report(y_true_cnn, y_pred_cnn, target_names=Config.TARGET_NAMES))

# Part II: DistilBERT Model

# 12. Check and import transformers
import transformers
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW

# 13. BERT Dataset definition
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
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 14. Load pre-trained model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
model_bert = DistilBertForSequenceClassification.from_pretrained(Config.BERT_MODEL_NAME, num_labels=len(Config.TARGET_NAMES))
model_bert.to(Config.DEVICE)

train_dataset_bert = BertDataset(X_train, y_train, tokenizer, max_len=Config.BERT_MAX_LEN)
test_dataset_bert = BertDataset(X_test, y_test, tokenizer, max_len=Config.BERT_MAX_LEN)
train_loader_bert = DataLoader(train_dataset_bert, batch_size=Config.BERT_BATCH_SIZE, shuffle=True)
test_loader_bert = DataLoader(test_dataset_bert, batch_size=Config.BERT_BATCH_SIZE, shuffle=False)

optimizer_bert = AdamW(model_bert.parameters(), lr=Config.BERT_LR)
criterion_bert = nn.CrossEntropyLoss(weight=torch.tensor(Config.CLASS_WEIGHTS).to(Config.DEVICE))

# 15. Training functions (BERT)
def train_bert_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(Config.DEVICE)
        attention_mask = batch['attention_mask'].to(Config.DEVICE)
        labels = batch['labels'].to(Config.DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_bert(model, loader, criterion):
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
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            _, pred = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return total_loss / len(loader), correct / total

# 16. Train BERT
print("\nTraining DistilBERT...")
for epoch in range(Config.BERT_EPOCHS):
    train_loss = train_bert_epoch(model_bert, train_loader_bert, optimizer_bert, criterion_bert)
    test_loss, test_acc = evaluate_bert(model_bert, test_loader_bert, criterion_bert)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

# 17. BERT classification report
def get_bert_predictions(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels_batch = batch['labels'].to(Config.DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, pred = torch.max(outputs.logits, 1)
            preds.extend(pred.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())
    return preds, labels

y_pred_bert, y_true_bert = get_bert_predictions(model_bert, test_loader_bert)
print("\nDistilBERT Classification Report:")
print(classification_report(y_true_bert, y_pred_bert, target_names=Config.TARGET_NAMES))

print("\nDone.")


