import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from config import Config

# Download stopwords and wordnet
# Uncomment these lines if you haven't downloaded them before
# nltk.download('stopwords')
# nltk.download('wordnet')

df = pd.read_csv(Config.RAW_DATA_PATH, header=None, names=['sentiment', 'text'], encoding='latin-1') # read raw data

def show_statistics(df, data_column, label_column): # show statistics of data
    print("Data shape:", df.shape)
    print("Label distribution:")
    print(df[label_column].value_counts())

    print("Text length statistics(character level):")
    print(df[data_column].apply(len).describe())

    print("Text length statistics(word level):")
    print(df[data_column].apply(lambda x: len(x.split())).describe())

def preprocess_text(text): # preprocess text
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

# Save processed data
df.to_csv(Config.PROCESSED_DATA_PATH, index=False)

# Sample data from each class to create train/test subset
train_df = df.groupby('sentiment').sample(frac=Config.TRAIN_RATIO, random_state=Config.SEED)
test_df = df.drop(train_df.index)

# Save train/test data
train_df.to_csv(Config.TRAIN_DATA_PATH, index=False)
test_df.to_csv(Config.TEST_DATA_PATH, index=False)
print(f"Train data saved to {Config.TRAIN_DATA_PATH}, size: {train_df.shape}")
print(f"Test data saved to {Config.TEST_DATA_PATH}, size: {test_df.shape}")
