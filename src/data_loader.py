# scripts/data_loader.py
import pandas as pd

def load_data(filepath):
    try:
        df = pd.read_csv(filepath, index_col=0)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None