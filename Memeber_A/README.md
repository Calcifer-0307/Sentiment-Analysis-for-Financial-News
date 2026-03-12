# What do we have
- This project offers PyTorch Dataloader for BoW, TF-IDF, and Word2Vec features.

# What is in each file

- `data_helper.py` : Contains function to build dataloader for BoW, TF-IDF, and Word2Vec features.
- `preprocess.py` : Contains function to preprocess text data.
- `data/all-data.csv`: Original dataset from Kaggle.
- `data/processed_data.csv`: Preprocessed dataset, a union of train and test dataset.
- `data/train_data.csv`: Preprocessed training dataset, takes 0.75 of the original dataset.
- `data/test_data.csv`: Preprocessed testing dataset, takes 0.25 of the original dataset.
- `total_report.txt`: A final report to conclude the data processing.
- `preprocess_report.txt`: A report of the preprocessing steps. It also includes the statistics of the dataset.

# How to use
`pip install -r requirements.txt`

```python
from data_helper import get_bow_dataloaders, get_tfidf_dataloaders, get_w2c_dataloaders, train_bow_or_tfidf_model, train_word2vec_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

train_data_path = 'data/train_data.csv'
test_data_path = 'data/test_data.csv'
feature_col = 'processed_text'
label_col = 'sentiment'

batch_size = 32
max_features = 5000     # Gives a batch data shape of (batch_size, max_features)
w2c_dim = 100           # Gives a batch data shape of (batch_size, w2c_dim). p.s: vector is the average of all words in the sentence.

# Train bow, tfidf, and word2vec models on the entire dataset to make sure that the vocabulary is consistent.
bow_model = train_bow_or_tfidf_model([train_data_path, test_data_path], CountVectorizer, feature_col=feature_col, max_features=max_features)`
tfidf_model = train_bow_or_tfidf_model([train_data_path, test_data_path], TfidfVectorizer, feature_col=feature_col, max_features=max_features)
w2c_model = train_word2vec_model([train_data_path, test_data_path], feature_col=feature_col, dim=w2c_dim)   

# Get dataloaders for each model
bow_train_loader, bow_test_loader = get_bow_dataloaders(train_data_path, test_data_path, bow_model, feature_col=feature_col, label_col=label_col, batch_size=batch_size)
tfidf_train_loader, tfidf_test_loader = get_tfidf_dataloaders(train_data_path, test_data_path, tfidf_model, feature_col=feature_col, label_col=label_col, batch_size=batch_size)    
w2c_train_loader, w2c_test_loader = get_w2c_dataloaders(train_data_path, test_data_path, w2c_model, feature_col=feature_col, label_col=label_col, batch_size=batch_size)
    
# Your code to enumerate dataloaders and train models....
```
