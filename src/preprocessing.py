"""
Preprocessing utilities for stock price prediction

This module provides reusable functions for data preprocessing including:
- Missing value handling
- Outlier detection and treatment
- Feature scaling
- Temporal train/test splits
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Optional
import joblib


def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'forward_fill',
                          columns: Optional[list] = None) -> pd.DataFrame:
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Strategy for handling missing values:
        - 'forward_fill': Forward fill (carry last known value)
        - 'backward_fill': Backward fill
        - 'interpolate': Linear interpolation
        - 'drop': Drop rows with missing values
    columns : list, optional
        Specific columns to apply strategy. If None, applies to all columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    if strategy == 'forward_fill':
        df_copy[columns] = df_copy[columns].fillna(method='ffill')
    elif strategy == 'backward_fill':
        df_copy[columns] = df_copy[columns].fillna(method='bfill')
    elif strategy == 'interpolate':
        df_copy[columns] = df_copy[columns].interpolate(method='linear')
    elif strategy == 'drop':
        df_copy = df_copy.dropna(subset=columns)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_copy


def detect_and_treat_outliers(df: pd.DataFrame,
                               columns: list,
                               method: str = 'clip',
                               lower_percentile: float = 1,
                               upper_percentile: float = 99) -> pd.DataFrame:
    """
    Detect and treat outliers using percentile-based clipping
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to check for outliers
    method : str
        Treatment method:
        - 'clip': Winsorization (clip values at percentiles)
        - 'remove': Remove outlier rows
    lower_percentile : float
        Lower percentile for clipping (default: 1)
    upper_percentile : float
        Upper percentile for clipping (default: 99)
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers treated
    """
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        lower_bound = df_copy[col].quantile(lower_percentile / 100)
        upper_bound = df_copy[col].quantile(upper_percentile / 100)
        
        if method == 'clip':
            df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'remove':
            df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return df_copy


def scale_features(df: pd.DataFrame,
                   features: list,
                   scaler_type: str = 'standard',
                   fit_on_train: bool = True,
                   scaler_path: Optional[str] = None) -> Tuple[pd.DataFrame, object]:
    """
    Scale features using specified scaler
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features : list
        List of feature columns to scale
    scaler_type : str
        Type of scaler:
        - 'standard': StandardScaler (mean=0, std=1)
        - 'minmax': MinMaxScaler (range [0, 1])
        - 'robust': RobustScaler (uses median and IQR)
    fit_on_train : bool
        If True, fit scaler on this data. If False, load existing scaler
    scaler_path : str, optional
        Path to save/load scaler
        
    Returns:
    --------
    Tuple[pd.DataFrame, object]
        Scaled dataframe and fitted scaler object
    """
    df_copy = df.copy()
    
    # Select scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Fit or load scaler
    if fit_on_train:
        scaler.fit(df_copy[features])
        if scaler_path:
            joblib.dump(scaler, scaler_path)
            print(f"Scaler saved to: {scaler_path}")
    else:
        if scaler_path:
            scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from: {scaler_path}")
        else:
            raise ValueError("scaler_path required when fit_on_train=False")
    
    # Transform features
    df_copy[features] = scaler.transform(df_copy[features])
    
    return df_copy, scaler


def temporal_train_test_split(df: pd.DataFrame,
                               train_end: str = '2018-12-31',
                               val_end: str = '2021-12-31',
                               date_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally for time series (NO SHUFFLING)
    
    CRITICAL: This function ensures no lookahead bias by splitting chronologically
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index or column
    train_end : str
        End date for training set (format: 'YYYY-MM-DD')
    val_end : str
        End date for validation set (format: 'YYYY-MM-DD')
    date_column : str, optional
        Name of date column if not using index
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train, validation, test dataframes
    """
    df_copy = df.copy()
    
    # Ensure datetime index
    if date_column and date_column in df_copy.columns:
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        df_copy.set_index(date_column, inplace=True)
    elif not isinstance(df_copy.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index or specify date_column")
    
    # Sort by date to ensure temporal order
    df_copy = df_copy.sort_index()
    
    # Split data
    train = df_copy[df_copy.index <= train_end]
    val = df_copy[(df_copy.index > train_end) & (df_copy.index <= val_end)]
    test = df_copy[df_copy.index > val_end]
    
    print(f"Train set: {train.index.min()} to {train.index.max()} ({len(train)} samples)")
    print(f"Validation set: {val.index.min()} to {val.index.max()} ({len(val)} samples)")
    print(f"Test set: {test.index.min()} to {test.index.max()} ({len(test)} samples)")
    
    # Verify no overlap
    assert train.index.max() < val.index.min(), "Train and validation sets overlap!"
    assert val.index.max() < test.index.min(), "Validation and test sets overlap!"
    
    return train, val, test


def create_sequences(data: np.ndarray, 
                     target: np.ndarray,
                     sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM/RNN models
    
    Parameters:
    -----------
    data : np.ndarray
        Feature array
    target : np.ndarray
        Target array
    sequence_length : int
        Length of lookback window
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        X (3D array: samples, timesteps, features) and y (1D array)
    """
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(target[i])
    
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Example usage
    print("Preprocessing utilities module")
    print("Import this module to use preprocessing functions")
