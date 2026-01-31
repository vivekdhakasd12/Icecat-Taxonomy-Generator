import pandas as pd
import os
from . import config

def load_icecat_data(filepath=config.DATA_PATH, max_rows=config.MAX_ROWS):
    """
    Loads the Icecat dataset from JSON.
    Handles large files by reading carefully.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    print(f"Loading data from {filepath}...")
    
    try:
        if max_rows:
             df = pd.read_json(filepath)
             if len(df) > max_rows:
                 print(f"Data large ({len(df)} rows). Sampling {max_rows} rows for analysis...")
                 df = df.sample(n=max_rows, random_state=config.RANDOM_SEED)
        else:
             df = pd.read_json(filepath)
             
    except ValueError:
        print("Standard load failed. Trying lines=True...")
        df = pd.read_json(filepath, lines=True)
        if max_rows and len(df) > max_rows:
             df = df.sample(n=max_rows, random_state=config.RANDOM_SEED)

    print(f"Data loaded. Shape: {df.shape}")
    return df
