"""
LightGBM Model Implementation for Stock Price Prediction

This module provides training and prediction functions for LightGBM regression/classification
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import joblib


def train_lightgbm(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_val: pd.DataFrame,
                   y_val: pd.Series,
                   objective: str = 'regression',
                   params: Optional[Dict] = None,
                   early_stopping_rounds: int = 50) -> Tuple[lgb.Booster, Dict]:
    """
    Train LightGBM model with early stopping
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
    objective : str
        LightGBM objective function:
        - 'regression' for regression
        - 'binary' for binary classification
        - 'multiclass' for multiclass
    params : dict, optional
        Custom parameters. If None, uses defaults
    early_stopping_rounds : int
        Stop if no improvement for N rounds
        
    Returns:
    --------
    Tuple[lgb.Booster, Dict]
        Trained model and training history
    """
    # Default parameters
    default_params = {
        'objective': objective,
        'metric': 'rmse' if objective == 'regression' else 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'seed': 42,
        'verbose': -1,
        'force_col_wise': True
    }
    
    # Update with custom params if provided
    if params:
        default_params.update(params)
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=X_train.columns.tolist())
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=X_val.columns.tolist())
    
    # Training
    print("Training LightGBM model...")
    print(f"Parameters: {default_params}")
    
    # In recent LightGBM versions, evaluation results are recorded via callbacks
    evals_result = {}
    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100),
            lgb.record_evaluation(evals_result)
        ],
    )
    
    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score['val'][default_params['metric']]:.6f}")
    
    return model, evals_result


def predict_lightgbm(model: lgb.Booster,
                     X_test: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using trained LightGBM model
    
    Parameters:
    -----------
    model : lgb.Booster
        Trained model
    X_test : pd.DataFrame
        Test features
        
    Returns:
    --------
    np.ndarray
        Predictions
    """
    predictions = model.predict(X_test, num_iteration=model.best_iteration)
    return predictions


def get_feature_importance(model: lgb.Booster,
                           importance_type: str = 'gain') -> pd.DataFrame:
    """
    Extract feature importance from LightGBM model
    
    Parameters:
    -----------
    model : lgb.Booster
        Trained model
    importance_type : str
        Type of importance: 'gain' or 'split'
        
    Returns:
    --------
    pd.DataFrame
        Feature importance sorted by importance
    """
    importance_df = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(importance_type=importance_type)
    }).sort_values('importance', ascending=False)
    
    return importance_df


def save_model(model: lgb.Booster, filepath: str):
    """Save LightGBM model to file"""
    model.save_model(filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath: str) -> lgb.Booster:
    """Load LightGBM model from file"""
    model = lgb.Booster(model_file=filepath)
    print(f"Model loaded from: {filepath}")
    return model


if __name__ == "__main__":
    print("LightGBM model implementation module")
    print("Import this module to train and use LightGBM models")
