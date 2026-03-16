# Financial News Sentiment Analysis

This project implements sentiment analysis on financial news using various feature extraction techniques (BoW, TF-IDF, Word2Vec) and PyTorch models.

## Project Structure

- `data/`: Contains raw and processed data.
  - `raw/`: Raw datasets.
  - `processed/`: Processed datasets ready for modeling.
- `src/`: Source code.
  - `preprocess.py`: Text cleaning and preprocessing script.
  - `data_helper.py`: PyTorch Dataset and DataLoader implementations.
- `notebooks/`: Jupyter notebooks for analysis.
- `requirements.txt`: Python dependencies.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Calcifer-0307/Sentiment-Analysis-for-Financial-News.git
cd Sentiment-Analysis-for-Financial-News
```

### 2. Set Up Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```cmd
python -m venv venv
venv\Scripts\activate
```

*(Note for Windows PowerShell users: If you encounter an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

The preprocessing script relies on NLTK data (stopwords, wordnet).

**Option A (Automatic):**
The script attempts to download necessary data automatically.

**Option B (Manual):**
If you encounter SSL errors or download failures, run the following python commands:

```python
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
```

## Usage

### 1. Data Preprocessing

Run the preprocessing script to clean the raw data and generate train/test splits.

```bash
# Ensure your virtual environment is activated
python src/preprocess.py
```

This will generate:
- `data/raw/train_data.csv`
- `data/raw/test_data.csv`

### 2. Loading Data for Training

You can use `src/data_helper.py` to create PyTorch DataLoaders.

```python
from src.data_helper import get_bow_dataloaders, get_tfidf_dataloaders, get_w2c_dataloaders
from src.data_helper import train_bow_or_tfidf_model, train_word2vec_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Configuration
train_path = 'data/raw/train_data.csv'
test_path = 'data/raw/test_data.csv'
batch_size = 32

# Example: BoW Model
# 1. Train Vectorizer
bow_model = train_bow_or_tfidf_model([train_path, test_path], CountVectorizer, max_features=5000)

# 2. Get DataLoaders
train_loader, test_loader = get_bow_dataloaders(
    train_path, test_path, 
    bow_model, 
    batch_size=batch_size
)

# 3. Iterate
for batch_x, batch_y in train_loader:
    # batch_x: (batch_size, 5000)
    # batch_y: (batch_size,)
    pass
```

## Troubleshooting

- **SSL Certificate Errors**: If you see SSL errors when downloading NLTK data on macOS, run `/Applications/Python 3.x/Install Certificates.command` or use the manual download script above.
- **ModuleNotFoundError**: Ensure you have activated your virtual environment (`source venv/bin/activate` or `venv\Scripts\activate`) before running scripts.
