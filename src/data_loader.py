import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import CSV_PATH
import pandas as pd

def load_data(path=None):
    if path is None:
        path = CSV_PATH
        
    if os.path.exists(path):
        df = pd.read_csv(path)
        print("Data loaded successfully!")
        return df
    else:
        print(f"Error: File not found at {path}")

if __name__ == "__main__":
    load_data()