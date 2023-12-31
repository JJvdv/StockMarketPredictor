import numpy as np
import pandas as pd

from yahoo_fin import stock_info as si
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import deque

'''
Preprocessing Dataset
'''
def shuffleInUnison(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)
    
def loadData(ticker, n_steps = 50, scale = True, shuffle = True, lookup_steps = 1, split_by_date = True, test_size = 0.2, 
             feature_columns = ['adjclose', 'volume', 'open', 'high', 'low']):
    if isinstance(ticker, str):
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instance")
        
    result = {}
    result['df'] = df.copy()
    
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    
    if "date" not in df.columns:
        df["date"] = df.index
        
    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
            
        result["column_scaler"] = column_scaler
        
    df['future'] = df['adjclose'].shift(-lookup_steps)
    last_sequence = np.array(df[feature_columns].tail(lookup_steps))
    df.dropna(inplace=True)
    
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    
    result['last_sequence'] = last_sequence
    
    X, y = [], []
    
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
        
    X = np.array(X)
    y = np.array(y)
    
    if split_by_date:
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_train"] = y[:train_samples]
        result["y_test"] = y[train_samples:]
        
        if shuffle:
            shuffleInUnison(result["X_train"], result["y_train"])
            shuffleInUnison(result["X_test"], result["y_test"])
    else:
        # Split data randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    
    dates = result["X_test"][:, -1, -1]
    
    result["test_df"] = result["df"].loc[dates]
    # Remove duplicates
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # Removes dates from training and testing set and convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    
    return result