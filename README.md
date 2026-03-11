# Sentiment Analysis for Financial News

This project aims to perform sentiment analysis on financial news articles.

## Project Structure

- `data/`: Contains raw, processed, and external data.
- `notebooks/`: Jupyter notebooks for data exploration and model prototyping.
- `src/`: Source code for the project.
  - `data/`: Scripts to fetch and preprocess data.
  - `features/`: Scripts to turn raw data into features for modeling.
  - `models/`: Scripts to train models and make predictions.
  - `visualization/`: Scripts to create visualizations.
  - `utils/`: Helper functions.
- `tests/`: Unit tests for the project.
- `main.py`: Main entry point for the application.

## Getting Started

### Prerequisites

- Python 3.8 or higher

### Installation

1. **Navigate to the project directory**.

2. **Create a virtual environment** (Recommended to isolate dependencies):

   It is best practice to use a virtual environment. Run the following commands based on your operating system:

   - **macOS / Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

   - **Windows** (Command Prompt):
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```

   - **Windows** (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install dependencies**:
   Once the virtual environment is activated, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Run the main script**:
   ```bash
   python main.py
   ```

2. **Run Tests**:
   To verify that everything is set up correctly, you can run the tests:
   ```bash
   pytest
   ```

## Notes for Windows Users

- **PowerShell Execution Policy**: If you encounter permission errors when activating the virtual environment in PowerShell, you may need to change the execution policy. Run the following command in PowerShell:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- **Path Issues**: Ensure that Python is added to your system's PATH variable so that you can run `python` from the command line.
