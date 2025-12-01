"""
LSTM Model Implementation for Stock Price Prediction

This module provides LSTM-based deep learning models for time series prediction
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler
import joblib


def create_lstm_model(input_shape: Tuple[int, int],
                      lstm_units: list = [128, 64],
                      dropout_rate: float = 0.2,
                      output_units: int = 1,
                      activation: str = 'linear') -> keras.Model:
    """
    Create LSTM model architecture
    
    Parameters:
    -----------
    input_shape : Tuple[int, int]
        (sequence_length, n_features)
    lstm_units : list
        Number of units in each LSTM layer
    dropout_rate : float
        Dropout rate for regularization
    output_units : int
        Number of output units (1 for regression, >1 for classification)
    activation : str
        Output activation: 'linear' for regression, 'sigmoid'/'softmax' for classification
        
    Returns:
    --------
    keras.Model
        Compiled LSTM model
    """
    model = keras.Sequential(name='LSTM_Stock_Predictor')
    
    # First LSTM layer (return sequences for stacked LSTM)
    model.add(layers.LSTM(
        units=lstm_units[0],
        return_sequences=len(lstm_units) > 1,
        input_shape=input_shape,
        name='LSTM_1'
    ))
    model.add(layers.Dropout(dropout_rate, name='Dropout_1'))
    
    # Additional LSTM layers
    for i, units in enumerate(lstm_units[1:], start=2):
        return_seq = i < len(lstm_units)  # Return sequences if not last LSTM layer
        model.add(layers.LSTM(
            units=units,
            return_sequences=return_seq,
            name=f'LSTM_{i}'
        ))
        model.add(layers.Dropout(dropout_rate, name=f'Dropout_{i}'))
    
    # Output layer
    model.add(layers.Dense(output_units, activation=activation, name='Output'))
    
    return model


def compile_model(model: keras.Model,
                  loss: str = 'mse',
                  learning_rate: float = 0.001,
                  metrics: list = None) -> keras.Model:
    """
    Compile LSTM model
    
    Parameters:
    -----------
    model : keras.Model
        Model to compile
    loss : str
        Loss function: 'mse', 'mae', 'binary_crossentropy', 'categorical_crossentropy'
    learning_rate : float
        Learning rate for optimizer
    metrics : list
        Metrics to track during training
        
    Returns:
    --------
    keras.Model
        Compiled model
    """
    if metrics is None:
        if 'mse' in loss or 'mae' in loss:
            metrics = ['mae', 'mse']
        else:
            metrics = ['accuracy']
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def prepare_lstm_data(X: np.ndarray,
                      y: np.ndarray,
                      sequence_length: int = 60,
                      scale: bool = True) -> Tuple:
    """
    Prepare data for LSTM (create sequences and scale)
    
    Parameters:
    -----------
    X : np.ndarray
        Features array
    y : np.ndarray
        Target array
    sequence_length : int
        Lookback window size
    scale : bool
        Whether to scale features
        
    Returns:
    --------
    Tuple
        (X_sequences, y_sequences, scaler_X, scaler_y)
    """
    scaler_X = MinMaxScaler() if scale else None
    scaler_y = MinMaxScaler() if scale else None
    
    # Scale data
    if scale:
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    else:
        X_scaled = X
        y_scaled = y
    
    # Create sequences
    X_seq, y_seq = [], []
    
    for i in range(sequence_length, len(X_scaled)):
        X_seq.append(X_scaled[i-sequence_length:i])
        y_seq.append(y_scaled[i])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"Sequence shape: {X_seq.shape}")
    print(f"Target shape: {y_seq.shape}")
    
    return X_seq, y_seq, scaler_X, scaler_y


def train_lstm(model: keras.Model,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: np.ndarray,
               y_val: np.ndarray,
               epochs: int = 100,
               batch_size: int = 32,
               verbose: int = 1) -> Tuple[keras.Model, Dict]:
    """
    Train LSTM model
    
    Parameters:
    -----------
    model : keras.Model
        Compiled model
    X_train : np.ndarray
        Training sequences (samples, timesteps, features)
    y_train : np.ndarray
        Training targets
    X_val : np.ndarray
        Validation sequences
    y_val : np.ndarray
        Validation targets
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    verbose : int
        Verbosity level
        
    Returns:
    --------
    Tuple[keras.Model, Dict]
        Trained model and training history
    """
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    print("Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose
    )
    
    return model, history.history


def predict_lstm(model: keras.Model,
                 X: np.ndarray,
                 scaler_y: Optional[MinMaxScaler] = None) -> np.ndarray:
    """
    Make predictions using trained LSTM model
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    X : np.ndarray
        Input sequences
    scaler_y : MinMaxScaler, optional
        Target scaler for inverse transform
        
    Returns:
    --------
    np.ndarray
        Predictions
    """
    predictions = model.predict(X, verbose=0)
    
    if scaler_y is not None:
        predictions = scaler_y.inverse_transform(predictions)
    
    return predictions.flatten()


def save_lstm_model(model: keras.Model,
                    scaler_X: Optional[MinMaxScaler],
                    scaler_y: Optional[MinMaxScaler],
                    model_path: str,
                    scaler_path: str):
    """
    Save LSTM model and scalers
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    scaler_X : MinMaxScaler
        Feature scaler
    scaler_y : MinMaxScaler
        Target scaler
    model_path : str
        Path to save model
    scaler_path : str
        Path to save scalers
    """
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    if scaler_X is not None and scaler_y is not None:
        joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, scaler_path)
        print(f"Scalers saved to: {scaler_path}")


def load_lstm_model(model_path: str,
                    scaler_path: Optional[str] = None) -> Tuple:
    """
    Load LSTM model and scalers
    
    Parameters:
    -----------
    model_path : str
        Path to saved model
    scaler_path : str, optional
        Path to saved scalers
        
    Returns:
    --------
    Tuple
        (model, scaler_X, scaler_y)
    """
    model = keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    
    if scaler_path:
        scalers = joblib.load(scaler_path)
        scaler_X = scalers['scaler_X']
        scaler_y = scalers['scaler_y']
        print(f"Scalers loaded from: {scaler_path}")
        return model, scaler_X, scaler_y
    
    return model, None, None


# Full training pipeline
def train_lstm_pipeline(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: pd.DataFrame,
                       y_val: pd.Series,
                       sequence_length: int = 60,
                       lstm_units: list = [128, 64],
                       dropout_rate: float = 0.2,
                       learning_rate: float = 0.001,
                       epochs: int = 100,
                       batch_size: int = 32) -> Tuple:
    """
    Complete LSTM training pipeline
    
    Returns:
    --------
    Tuple
        (model, history, scaler_X, scaler_y)
    """
    # Prepare data
    X_train_seq, y_train_seq, scaler_X, scaler_y = prepare_lstm_data(
        X_train.values, y_train.values, sequence_length, scale=True
    )
    
    X_val_seq, y_val_seq, _, _ = prepare_lstm_data(
        X_val.values, y_val.values, sequence_length, scale=False
    )
    
    # Scale validation data using training scalers
    X_val_seq = scaler_X.transform(X_val_seq.reshape(-1, X_val_seq.shape[-1])).reshape(X_val_seq.shape)
    y_val_seq = scaler_y.transform(y_val_seq.reshape(-1, 1)).flatten()
    
    # Create and compile model
    input_shape = (sequence_length, X_train.shape[1])
    model = create_lstm_model(
        input_shape=input_shape,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate
    )
    
    model = compile_model(model, learning_rate=learning_rate)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    model, history = train_lstm(
        model, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        epochs=epochs, batch_size=batch_size
    )
    
    return model, history, scaler_X, scaler_y


if __name__ == "__main__":
    print("LSTM model implementation module")
    print("Import this module to train and use LSTM models")
