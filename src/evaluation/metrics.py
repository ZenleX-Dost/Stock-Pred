"""
Evaluation Metrics for Stock Price Prediction

This module provides comprehensive evaluation metrics including:
- Regression metrics (MAE, RMSE, R², MAPE)
- Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Financial metrics (Sharpe Ratio, Max Drawdown, Win Rate)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from typing import Dict, Tuple


# ==================== REGRESSION METRICS ====================

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error"""
    return mean_squared_error(y_true, y_pred)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² Score"""
    return r2_score(y_true, y_pred)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error"""
    return mean_absolute_percentage_error(y_true, y_pred) * 100


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (% of correct up/down predictions)
    
    This is critical for trading strategies - even if price prediction is off,
    getting the direction right is what matters for profitability
    """
    actual_direction = np.diff(y_true) > 0
    predicted_direction = np.diff(y_pred) > 0
    return np.mean(actual_direction == predicted_direction) * 100


def regression_metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Generate comprehensive regression metrics report"""
    return {
        'MAE': calculate_mae(y_true, y_pred),
        'MSE': calculate_mse(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'Directional_Accuracy': calculate_directional_accuracy(y_true, y_pred)
    }


# ==================== CLASSIFICATION METRICS ====================

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score"""
    return accuracy_score(y_true, y_pred) * 100


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
    """Calculate precision"""
    return precision_score(y_true, y_pred, average=average, zero_division=0) * 100


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
    """Calculate recall"""
    return recall_score(y_true, y_pred, average=average, zero_division=0) * 100


def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
    """Calculate F1 score"""
    return f1_score(y_true, y_pred, average=average, zero_division=0) * 100


def calculate_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculate ROC-AUC score (requires predicted probabilities)"""
    try:
        return roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        return np.nan


def classification_metrics_report(y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """Generate comprehensive classification metrics report"""
    metrics = {
        'Accuracy': calculate_accuracy(y_true, y_pred),
        'Precision': calculate_precision(y_true, y_pred),
        'Recall': calculate_recall(y_true, y_pred),
        'F1': calculate_f1(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['ROC_AUC'] = calculate_roc_auc(y_true, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['Confusion_Matrix'] = cm.tolist()
    
    return metrics


# ==================== FINANCIAL METRICS ====================

def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate returns from prices"""
    return np.diff(prices) / prices[:-1]


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe Ratio (risk-adjusted returns)
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    risk_free_rate : float
        Annual risk-free rate (default: 2%)
        
    Returns:
    --------
    float
        Annualized Sharpe Ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    # Annualize returns (assuming daily data)
    excess_returns = returns - (risk_free_rate / 252)
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from equity curve
    
    Returns:
    --------
    Tuple[float, int, int]
        (max_drawdown_percentage, peak_index, trough_index)
    """
    cumulative_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    max_dd = np.min(drawdown) * 100  # Convert to percentage
    
    max_dd_idx = np.argmin(drawdown)
    peak_idx = np.argmax(equity_curve[:max_dd_idx]) if max_dd_idx > 0 else 0
    
    return max_dd, peak_idx, max_dd_idx


def calculate_win_rate(returns: np.ndarray) -> float:
    """Calculate percentage of profitable trades"""
    if len(returns) == 0:
        return 0.0
    return np.sum(returns > 0) / len(returns) * 100


def calculate_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (gross profit / gross loss)
    
    A profit factor > 1 indicates profitability
    """
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray) -> float:
    """
    Calculate Calmar Ratio (annual return / max drawdown)
    
    Measures return relative to downside risk
    """
    annual_return = np.mean(returns) * 252  # Annualized
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    
    if max_dd == 0:
        return 0.0
    
    return (annual_return / abs(max_dd)) * 100


def financial_metrics_report(returns: np.ndarray, 
                             equity_curve: np.ndarray,
                             initial_capital: float = 10000) -> Dict[str, float]:
    """
    Generate comprehensive financial performance report
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of trade returns
    equity_curve : np.ndarray
        Equity curve over time
    initial_capital : float
        Starting capital
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of financial metrics
    """
    final_capital = equity_curve[-1] if len(equity_curve) > 0 else initial_capital
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    
    max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity_curve)
    
    return {
        'Total_Return_%': total_return,
        'Final_Capital': final_capital,
        'Sharpe_Ratio': calculate_sharpe_ratio(returns),
        'Max_Drawdown_%': max_dd,
        'Win_Rate_%': calculate_win_rate(returns),
        'Profit_Factor': calculate_profit_factor(returns),
        'Calmar_Ratio': calculate_calmar_ratio(returns, equity_curve),
        'Total_Trades': len(returns),
        'Avg_Trade_Return_%': np.mean(returns) * 100 if len(returns) > 0 else 0.0
    }


if __name__ == "__main__":
    print("Evaluation metrics module")
    print("Provides comprehensive evaluation for regression, classification, and financial performance")
