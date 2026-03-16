import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

def train_bow_or_tfidf_model(data_paths, model_class, feature_col='processed_text', max_features=5000):
    """
        Why do we need this function: 
        the BoW model has to be trained on the entire dataset, 
        including training and testing, in order to ensure that the vocabulary is complete and consistent
        across both sets.
        
        model_class: CountVectorizer or TfidfVectorizer
    """
    dfs = [pd.read_csv(data_path) for data_path in data_paths]
    df = pd.concat(dfs, ignore_index=True)
    # drop rows with nan values in feature_col or label_col
    df = df.dropna(subset=[feature_col])
    
    texts = df[feature_col].tolist()
    # fit model
    model = model_class(max_features=max_features)
    model.fit(texts)
    
    return model

def train_word2vec_model(data_paths, feature_col='processed_text', dim=100):
    dfs = [pd.read_csv(data_path) for data_path in data_paths]
    df = pd.concat(dfs, ignore_index=True)
    # drop rows with nan values in feature_col or label_col
    df = df.dropna(subset=[feature_col])
    
    texts = df[feature_col].tolist()
    # fit model
    model = Word2Vec(sentences=texts, vector_size=dim)
    
    return model

class BaseDataset(Dataset):
    def __init__(self, data_path, feature_col='processed_text', label_col='sentiment'):
        # load processed data
        # drop rows with NaN values in feature_col or label_col
        self.df = pd.read_csv(data_path).dropna(subset=[feature_col, label_col])
        self.texts = self.df[feature_col].tolist()
        
        self.label_code_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.labels = [self.label_code_map[label] for label in self.df[label_col].tolist()]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

class BoWDataset(BaseDataset):
    def __init__(self, data_path, model, feature_col='processed_text', label_col='sentiment'):
        super().__init__(data_path, feature_col, label_col) 
        self.model = model
    
    def __getitem__(self, idx):
        text, label = super().__getitem__(idx)
        # calculate feature
        feat = self.model.transform([text]).toarray()[0]
        # convert to tensor
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

class TFIDFDataset(BaseDataset):
    def __init__(self, data_path, model, feature_col='processed_text', label_col='sentiment'):
        super().__init__(data_path, feature_col, label_col)     
        self.model = model
    
    def __getitem__(self, idx):
        text, label = super().__getitem__(idx)
        # calculate feature
        feat = self.model.transform([text]).toarray()[0]
        # convert to tensor
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

class W2CDataset(BaseDataset):
    def __init__(self, data_path, model, feature_col='processed_text', label_col='sentiment', dim=100):
        super().__init__(data_path, feature_col, label_col) 
        self.dim = dim
        self.model = model
    
    def __getitem__(self, idx):
        text, label = super().__getitem__(idx)
        # use the average of word vectors of the sentence to represent the sentence
        vectors = [self.model.wv[tok] for tok in text.split(" ") if tok in self.model.wv]
        if vectors:
            avg_vector = np.mean(vectors, axis=0)
        else:
            # use zero vector if tokens are not in the vocabulary
            avg_vector = np.zeros(self.dim)
        
        return torch.tensor(avg_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_bow_dataloaders(train_data_path, test_data_path, model, feature_col='processed_text', label_col='sentiment', 
                       batch_size=32, max_features=5000, shuffle=True):
    
    train_dataset = BoWDataset(train_data_path, model, feature_col=feature_col, label_col=label_col)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    test_dataset = BoWDataset(test_data_path, model, feature_col=feature_col, label_col=label_col)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, test_dataloader
    
def get_tfidf_dataloaders(train_data_path, test_data_path, model, feature_col='processed_text', label_col='sentiment', 
                       batch_size=32, max_features=5000, shuffle=True):
    
    train_dataset = TFIDFDataset(train_data_path, model, feature_col=feature_col, label_col=label_col)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    test_dataset = TFIDFDataset(test_data_path, model, feature_col=feature_col, label_col=label_col)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, test_dataloader
    
def get_w2c_dataloaders(train_data_path, test_data_path, model, feature_col='processed_text', label_col='sentiment', 
                       batch_size=32, dim=100, shuffle=True):
    
    train_dataset = W2CDataset(train_data_path, model, feature_col=feature_col, label_col=label_col, dim=dim)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    test_dataset = W2CDataset(test_data_path, model, feature_col=feature_col, label_col=label_col, dim=dim)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, test_dataloader
    
# Tests
if __name__ == "__main__":
    # 更新数据路径为data/raw目录下的文件
    train_data_path = 'data/processed/train_data.csv'
    test_data_path = 'data/processed/test_data.csv'
    feature_col = 'processed_text'
    label_col = 'sentiment'
    
    bow_model = train_bow_or_tfidf_model([train_data_path, test_data_path], CountVectorizer, feature_col=feature_col)
    tfidf_model = train_bow_or_tfidf_model([train_data_path, test_data_path], TfidfVectorizer, feature_col=feature_col)
    w2c_model = train_word2vec_model([train_data_path, test_data_path], feature_col=feature_col)
    
    bow_train_loader, bow_test_loader = get_bow_dataloaders(train_data_path, test_data_path, bow_model, feature_col=feature_col, label_col=label_col, batch_size=16)
    tfidf_train_loader, tfidf_test_loader = get_tfidf_dataloaders(train_data_path, test_data_path, tfidf_model, feature_col=feature_col, label_col=label_col, batch_size=16)
    w2c_train_loader, w2c_test_loader = get_w2c_dataloaders(train_data_path, test_data_path, w2c_model, feature_col=feature_col, label_col=label_col, batch_size=16)
    
    for dataloader in [bow_train_loader, bow_test_loader, tfidf_train_loader, tfidf_test_loader, w2c_train_loader, w2c_test_loader]:
        for batch_idx, (batch_data, labels) in enumerate(dataloader):
            print(f"{batch_idx}, feat shape: {batch_data.shape}, labels: {labels}")
            if batch_idx == 1:
                break