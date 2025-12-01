"""
XGBoost Model Implementation for Stock Price Prediction

This module provides training and prediction functions for XGBoost regression/classification
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import joblib


def train_xgboost(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_val: pd.DataFrame,
                  y_val: pd.Series,
                  objective: str = 'reg:squarederror',
                  params: Optional[Dict] = None,
                  early_stopping_rounds: int = 50) -> Tuple[xgb.Booster, Dict]:
    """
    Train XGBoost model with early stopping
    
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
        XGBoost objective function:
        - 'reg:squarederror' for regression
        - 'binary:logistic' for binary classification
        - 'multi:softmax' for multiclass
    params : dict, optional
        Custom parameters. If None, uses defaults
    early_stopping_rounds : int
        Stop if no improvement for N rounds
        
    Returns:
    --------
    Tuple[xgb.Booster, Dict]
        Trained model and training history
    """
    # Default parameters
    default_params = {
        'objective': objective,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'eval_metric': 'rmse' if 'reg' in objective else 'logloss',
        'seed': 42,
        'tree_method': 'hist'
    }
    
    # Update with custom params if provided
    if params:
        default_params.update(params)
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())
    
    # Watchlist for monitoring
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    
    # Training
    print("Training XGBoost model...")
    print(f"Parameters: {default_params}")
    
    evals_result = {}
    model = xgb.train(
        default_params,
        dtrain,
        num_boost_round=2000,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=100
    )
    
    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score:.6f}")
    
    return model, evals_result


def predict_xgboost(model: xgb.Booster,
                    X_test: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using trained XGBoost model
    
    Parameters:
    -----------
    model : xgb.Booster
        Trained model
    X_test : pd.DataFrame
        Test features
        
    Returns:
    --------
    np.ndarray
        Predictions
    """
    dtest = xgb.DMatrix(X_test, feature_names=X_test.columns.tolist())
    predictions = model.predict(dtest)
    return predictions


def get_feature_importance(model: xgb.Booster,
                           feature_names: list,
                           importance_type: str = 'gain') -> pd.DataFrame:
    """
    Extract feature importance from XGBoost model
    
    Parameters:
    -----------
    model : xgb.Booster
        Trained model
    feature_names : list
        List of feature names
    importance_type : str
        Type of importance: 'gain', 'weight', 'cover'
        
    Returns:
    --------
    pd.DataFrame
        Feature importance sorted by importance
    """
    importance_dict = model.get_score(importance_type=importance_type)
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    }).sort_values('importance', ascending=False)
    
    return importance_df


def save_model(model: xgb.Booster, filepath: str):
    """Save XGBoost model to file"""
    model.save_model(filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath: str) -> xgb.Booster:
    """Load XGBoost model from file"""
    model = xgb.Booster()
    model.load_model(filepath)
    print(f"Model loaded from: {filepath}")
    return model


if __name__ == "__main__":
    print("XGBoost model implementation module")
    print("Import this module to train and use XGBoost models")
