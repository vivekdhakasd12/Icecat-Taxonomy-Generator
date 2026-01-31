import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from . import config

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()

def create_text_features(df, text_cols=config.TEXT_COLS):
    """
    Concatenates text columns into a single 'cluster_text' feature.
    """
    print(f"Creating features from: {text_cols}")
    df['cluster_text'] = ""
    for col in text_cols:
        if col in df.columns:
            df['cluster_text'] += df[col].astype(str).fillna("") + " "
    
    df['cluster_text'] = df['cluster_text'].apply(clean_text)
    
    initial_len = len(df)
    df = df[df['cluster_text'].str.len() > 2].copy()
    print(f"Rows after cleaning: {len(df)} (Dropped {initial_len - len(df)})")
    return df

def generate_embeddings(df, model_name=config.EMBEDDING_MODEL):
    """
    Generates or loads embeddings for the 'cluster_text' column.
    """
    cache_file = os.path.join(config.CACHE_DIR, f"embeddings_{model_name}_{len(df)}.npy")
    
    if os.path.exists(cache_file):
        print(f"Loading embeddings from cache: {cache_file}")
        embeddings = np.load(cache_file)
        if len(embeddings) == len(df):
            return embeddings
        else:
            print("Cache size mismatch. Recomputing...")

    print(f"Encoding {len(df)} rows with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df['cluster_text'].tolist(), show_progress_bar=True, batch_size=64)
    

    print(f"Saving embeddings to: {cache_file}")
    np.save(cache_file, embeddings)
    
    return embeddings
