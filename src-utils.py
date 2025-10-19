import torch
import numpy as np
import random
import pandas as pd

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_obv(close, volume):
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    tr = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def load_data(eth_path='data/ethereum.csv', btc_path='data/btc_close.csv'):
    eth_data = pd.read_csv(eth_path)
    btc_data = pd.read_csv(btc_path)
    df = pd.DataFrame(index=eth_data.index)
    df['eth_open'] = eth_data['Open']
    df['eth_high'] = eth_data['High']
    df['eth_low'] = eth_data['Low']
    df['eth_volume'] = eth_data['Volume']
    df['eth_close'] = eth_data['Close']
    df['btc_close'] = btc_data['BTC_Close']
    df['obv'] = calculate_obv(df['eth_close'], df['eth_volume'])
    df['atr'] = calculate_atr(df['eth_high'], df['eth_low'], df['eth_close'])
    df.dropna(inplace=True)
    return df

def create_sequences(features, target, window_size):
    X_seq, y_seq = [], []
    for i in range(len(features) - window_size):
        X_seq.append(features[i: i + window_size])
        y_seq.append(target[i + window_size])
    return np.array(X_seq), np.array(y_seq)
