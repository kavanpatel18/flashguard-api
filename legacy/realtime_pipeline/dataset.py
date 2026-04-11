import os
import time
import requests
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepLOB_Dataset")

def fetch_binance_lob(symbol="BTCUSDT", limit=10, num_snapshots=500, delay_ms=50):
    """
    Scrapes real LOB data from Binance REST API to build a DeepLOB-compatible dataset.
    """
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
    data = []
    
    logger.info(f"Gathering {num_snapshots} real LOB snapshots from Binance for {symbol}...")
    for i in range(num_snapshots):
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                book = res.json()
                bids = book['bids']
                asks = book['asks']
                
                # DeepLOB format: [Pa1, Va1, Pb1, Vb1, Pa2, Va2, Pb2, Vb2...] (40 cols)
                row = []
                for lvl in range(limit):
                    if lvl < len(asks) and lvl < len(bids):
                        row.extend([
                            float(asks[lvl][0]), float(asks[lvl][1]), # Ask Price, Vol
                            float(bids[lvl][0]), float(bids[lvl][1])  # Bid Price, Vol
                        ])
                    else:
                        row.extend([0.0, 0.0, 0.0, 0.0])
                data.append(row)
            else:
                logger.warning(f"Failed to fetch data: {res.status_code}")
        except Exception as e:
            logger.error(f"Error fetching LOB: {e}")
            
        time.sleep(delay_ms / 1000.0)
        if (i+1) % 100 == 0:
            logger.info(f"Collected {i+1}/{num_snapshots} snapshots.")
            
    df = pd.DataFrame(data)
    df.to_csv("real_lob_data.csv", index=False)
    return df

def prepare_y(df, k=10, threshold=0.0001):
    """
    Computes binary Flash Crash Label
    1: Crash (Sharp Downwards movement), 0: Stationary/Up
    """
    # Mid-price: (Pa1 + Pb1) / 2
    # Pa1 is col 0, Pb1 is col 2
    mid_prices = (df.iloc[:, 0] + df.iloc[:, 2]) / 2.0
    
    # Calculate rolling min of future k mid-prices
    future_mins = mid_prices.rolling(window=k).min().shift(-k)
    
    returns = (future_mins - mid_prices) / mid_prices
    
    labels = np.zeros(len(df))
    # Label is 1 if it drops sharply beyond the negative threshold
    labels[returns < -threshold] = 1.0 
    return labels

def create_deeplob_tensors(df, seq_len=60):
    """
    Replicates the DeepLOB github methodology exactly:
    - Data Z-score normalization strictly computed on the Train split to avoid forward-looking bias.
    - Slicing data into sliding windows of seq_len.
    """
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx].values
    
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    std[std == 0] = 1.0 # prevent div by zero
    
    normalized_data = (df.values - mean) / std
    labels = prepare_y(df)
    
    X, y = [], []
    for i in range(len(normalized_data) - seq_len - 10): # 10 is k for future smoothing
        X.append(normalized_data[i : i + seq_len])
        y.append(labels[i + seq_len - 1])
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    return torch.tensor(X), torch.tensor(y), split_idx

def get_dataloaders(num_snapshots=500, seq_len=60, batch_size=32):
    if not os.path.exists("real_lob_data.csv"):
        df = fetch_binance_lob(num_snapshots=num_snapshots)
    else:
        df = pd.read_csv("real_lob_data.csv")
        
    # We add 5 dummy columns for ML-HFT features to match the config's num_features=45 parameter
    for i in range(5):
        df[f'extra_{i}'] = 0.0
        
    X_tensor, y_tensor, split_idx = create_deeplob_tensors(df, seq_len)
    
    train_size = int(len(X_tensor) * 0.8)
    X_train, y_train = X_tensor[:train_size], y_tensor[:train_size]
    X_val, y_val = X_tensor[train_size:], y_tensor[train_size:]
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
