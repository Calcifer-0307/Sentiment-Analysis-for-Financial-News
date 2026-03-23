# Financial News Sentiment Analysis

This project implements sentiment analysis on financial news using various feature extraction techniques (BoW, TF-IDF, Word2Vec) and PyTorch models (TextCNN and DistilBERT).

## Project Structure

- `data/`: Contains raw and processed data.
  - `raw/`: Raw datasets and preprocessed csv files.
- `src/`: Source code.
  - `config.py`: Centralized configuration file for hyperparameter tuning and path management.
  - `preprocess.py`: Text cleaning and train/test splitting script.
  - `data_helper.py`: Helper functions for BoW, TF-IDF, and Word2Vec models.
  - `train_CNN.py`: Training and evaluation script for the TextCNN model.
  - `train_transformer.py`: Training and evaluation script for the DistilBERT model.
- `notebooks/`: Text reports and previous jupyter notebooks.
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
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

*(Note for Windows PowerShell users: If you encounter an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

Run the preprocessing script to clean the raw data and generate train/test splits. It will automatically handle NLTK data downloads and SSL certificates.

```bash
# Ensure your virtual environment is activated
python src/preprocess.py
```

This will generate `train_data.csv` and `test_data.csv` in the `data/raw/` directory.

### 2. Training the Models

We have implemented two deep learning models for comparison. You can adjust hyperparameters (like learning rate, batch size, epochs) centrally in `src/config.py`.

**Run TextCNN Model:**
```bash
python src/train_CNN.py
```

**Run DistilBERT Model:**
```bash
python src/train_transformer.py
```

Both scripts will automatically load the data, train the model, and print out a detailed classification report containing Precision, Recall, and F1-score for each class (positive, negative, neutral).

## Configuration Management

All important paths, model hyperparameters, and data settings are stored in `src/config.py`. If you want to run comparative experiments (e.g., changing batch size or learning rate), please modify the corresponding variables in `Config` class:

```python
class Config:
    # Example hyperparameters
    CNN_LR = 1e-3
    CNN_BATCH_SIZE = 64
    
    BERT_LR = 2e-5
    BERT_BATCH_SIZE = 16
```

## Troubleshooting

- **SSL Certificate Errors**: The scripts now handle SSL errors automatically. However, if you still face issues downloading NLTK data on macOS, try running `/Applications/Python 3.x/Install Certificates.command`.
- **ModuleNotFoundError**: Ensure you have activated your virtual environment (`source .venv/bin/activate`) before running any scripts.
