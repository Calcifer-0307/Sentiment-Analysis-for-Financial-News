import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download stopwords and wordnet
# Uncomment these lines if you haven't downloaded them before
# nltk.download('stopwords')
# nltk.download('wordnet')

df = pd.read_csv('data/all-data.csv', header=None, names=['sentiment', 'text'], encoding='latin-1')

def show_statistics(df, data_column, label_column):
    print("Data shape:", df.shape)
    print("Label distribution:")
    print(df[label_column].value_counts())

    print("Text length statistics(character level):")
    print(df[data_column].apply(len).describe())

    print("Text length statistics(word level):")
    print(df[data_column].apply(lambda x: len(x.split())).describe())

def preprocess_text(text):
    # convert to lower case
    text = text.lower()
    # remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    tokens = text.split()
    # remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Show statistics before preprocessing
show_statistics(df, 'text', 'sentiment')
# Preprocess data
df['processed_text'] = df['text'].apply(preprocess_text)
# Show statistics after preprocessing
show_statistics(df, 'processed_text', 'sentiment')

df.to_csv('data/processed_data.csv', index=False)

# Sample data from each class to create train/test subset
train_ratio = 0.75
train_df = df.groupby('sentiment').sample(frac=train_ratio, random_state=42)
test_df = df.drop(train_df.index)

train_df_name = 'data/train_data.csv'
test_df_name = 'data/test_data.csv'

train_df.to_csv(train_df_name, index=False)
test_df.to_csv(test_df_name, index=False)
print(f"Train data saved to {train_df_name}, size: {train_df.shape}")
print(f"Test data saved to {test_df_name}, size: {test_df.shape}")
