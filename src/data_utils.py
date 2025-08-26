"""
data_utils.py
Utilities for loading Numerai data, preprocessing, and splitting by era.
"""

import pandas as pd
import numpy as np
from numerapi import NumerAPI
import os

def download_numerai_data(version="v4.3", dest_folder="data/raw"):
    """Download Numerai train.parquet to raw data folder."""
    os.makedirs(dest_folder, exist_ok=True)
    file_path = os.path.join(dest_folder, "train.parquet")
    napi = NumerAPI()
    napi.download_dataset(f"{version}/train.parquet", file_path)
    print(f"Downloaded Numerai data to {file_path}")

def load_data(file_path="data/raw/train.parquet"):
    df = pd.read_parquet(file_path)
    print(f"Loaded data shape: {df.shape}")
    return df

def preprocess_data(df):
    # Fill missing values, optionally normalize features
    df = df.fillna(df.mean())
    # Add more preprocessing as needed
    print("Preprocessed data.")
    return df

def split_by_era(df, val_frac=0.2):
    eras = df["era"].unique()
    val_eras = np.random.choice(eras, int(len(eras)*val_frac), replace=False)
    train = df[~df["era"].isin(val_eras)]
    val = df[df["era"].isin(val_eras)]
    print(f"Train eras: {len(train)}; Val eras: {len(val)}")
    return train, val