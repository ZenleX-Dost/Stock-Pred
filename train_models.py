"""
Model Training Script - Train XGBoost and LightGBM on entire dataset

This script trains both XGBoost and LightGBM models on the complete engineered
features dataset and saves them for use in the Streamlit application.

Run: python train_models.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models.xgboost_model import train_xgboost, save_model as save_xgb
from models.lightgbm_model import train_lightgbm, save_model as save_lgb
from preprocessing import handle_missing_values, scale_features


def load_data():
    """Load engineered features"""
    try:
        df = pd.read_csv('data/processed/features_engineered.csv', parse_dates=['dt'])
        df.set_index('dt', inplace=True)
        print(f"Loaded engineered features: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading engineered features: {e}")
        print("Please ensure feature engineering notebook has been run first.")
        return None


def prepare_data(df):
    """Prepare data for training"""
    print("\n[PREPROCESSING] Preparing data...")
    
    # Identify target and feature columns
    target_columns = [col for col in df.columns if 'target' in col]
    feature_cols = [col for col in df.columns if col not in target_columns and col != 'sp500']
    
    if not target_columns:
        print("ERROR: No target columns found in dataset!")
        return None, None, None, None
    
    # Use first target column (usually target_next_day)
    target_col = target_columns[0]
    print(f"Target column: {target_col}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Extract X and y
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove rows with missing target
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"Valid samples (non-null targets): {len(X)}")
    
    # Handle missing values in features
    X = handle_missing_values(X, strategy='forward_fill')
    
    # Check for any remaining NaNs
    if X.isnull().sum().sum() > 0:
        print(f"Remaining NaNs in features: {X.isnull().sum().sum()}")
        X = X.fillna(X.mean())
    
    print(f"Final dataset shape - X: {X.shape}, y: {y.shape}")
    print(f"Target statistics - Mean: {y.mean():.4f}, Std: {y.std():.4f}")
    
    return X, y, feature_cols, target_col


def train_and_save_models(X, y, feature_cols):
    """Train both models and save them"""
    
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL ON ENTIRE DATASET")
    print("="*80)
    
    # For XGBoost, use 80/20 train/val split to enable early stopping during training
    split_idx = int(0.8 * len(X))
    X_train_xgb = X.iloc[:split_idx]
    y_train_xgb = y.iloc[:split_idx]
    X_val_xgb = X.iloc[split_idx:]
    y_val_xgb = y.iloc[split_idx:]
    
    print(f"XGBoost Training set: {X_train_xgb.shape}")
    print(f"XGBoost Validation set: {X_val_xgb.shape}")
    
    try:
        xgb_model, xgb_evals = train_xgboost(
            X_train_xgb, y_train_xgb,
            X_val_xgb, y_val_xgb,
            objective='reg:squarederror',
            params={
                'learning_rate': 0.05,
                'max_depth': 7,
                'min_child_weight': 1,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'gamma': 1,
                'reg_alpha': 0.5,
                'reg_lambda': 1.5
            },
            early_stopping_rounds=50
        )
        
        save_xgb(xgb_model, 'models/xgboost_next_day.json')
        print("XGBoost model saved successfully!")
        
    except Exception as e:
        print(f"ERROR training XGBoost: {e}")
        import traceback
        traceback.print_exc()
        xgb_model = None
    
    print("\n" + "="*80)
    print("TRAINING LIGHTGBM MODEL ON ENTIRE DATASET")
    print("="*80)
    
    # For LightGBM, use 80/20 train/val split
    X_train_lgb = X.iloc[:split_idx]
    y_train_lgb = y.iloc[:split_idx]
    X_val_lgb = X.iloc[split_idx:]
    y_val_lgb = y.iloc[split_idx:]
    
    print(f"LightGBM Training set: {X_train_lgb.shape}")
    print(f"LightGBM Validation set: {X_val_lgb.shape}")
    
    try:
        lgb_model, lgb_evals = train_lightgbm(
            X_train_lgb, y_train_lgb,
            X_val_lgb, y_val_lgb,
            objective='regression',
            params={
                'learning_rate': 0.05,
                'num_leaves': 127,
                'max_depth': 8,
                'min_child_samples': 10,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.9,
                'bagging_freq': 5,
                'reg_alpha': 0.5,
                'reg_lambda': 1.5
            },
            early_stopping_rounds=50
        )
        
        save_lgb(lgb_model, 'models/lightgbm_next_day.txt')
        print("LightGBM model saved successfully!")
        
    except Exception as e:
        print(f"ERROR training LightGBM: {e}")
        import traceback
        traceback.print_exc()
        lgb_model = None
    
    return xgb_model, lgb_model


def evaluate_models(X, y, xgb_model, lgb_model):
    """Quick evaluation of trained models"""
    from evaluation.metrics import regression_metrics_report
    from models.xgboost_model import predict_xgboost
    from models.lightgbm_model import predict_lightgbm
    
    print("\n" + "="*80)
    print("MODEL EVALUATION ON TEST SET (Last 20% of data)")
    print("="*80)
    
    # Test set (last 20%)
    split_idx = int(0.8 * len(X))
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    print(f"\nTest set size: {X_test.shape[0]} samples")
    
    if xgb_model is not None:
        print("\n--- XGBOOST METRICS ---")
        xgb_pred = predict_xgboost(xgb_model, X_test)
        xgb_metrics = regression_metrics_report(y_test.values, xgb_pred)
        
        for metric, value in xgb_metrics.items():
            if metric != 'Confusion_Matrix':
                print(f"{metric}: {value:.6f}")
    
    if lgb_model is not None:
        print("\n--- LIGHTGBM METRICS ---")
        lgb_pred = predict_lightgbm(lgb_model, X_test)
        lgb_metrics = regression_metrics_report(y_test.values, lgb_pred)
        
        for metric, value in lgb_metrics.items():
            if metric != 'Confusion_Matrix':
                print(f"{metric}: {value:.6f}")


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("STOCK PRICE PREDICTION - MODEL TRAINING SCRIPT")
    print("="*80)
    print(f"Training on: ENTIRE ENGINEERED FEATURES DATASET")
    print("="*80)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare data
    X, y, feature_cols, target_col = prepare_data(df)
    if X is None:
        return
    
    # Train and save models
    xgb_model, lgb_model = train_and_save_models(X, y, feature_cols)
    
    # Evaluate
    if xgb_model is not None or lgb_model is not None:
        evaluate_models(X, y, xgb_model, lgb_model)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nModels saved to:")
    print("  - models/xgboost_next_day.json")
    print("  - models/lightgbm_next_day.txt")
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run app.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
