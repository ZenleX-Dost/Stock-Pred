"""
Trading Strategy Backtesting

This module simulates trading based on model predictions and calculates
performance metrics including transaction costs
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime


class TradingBacktester:
    """
    Backtest trading strategies with realistic constraints
    
    Features:
    - Transaction costs
    - Slippage simulation
    - Position sizing
    - Risk management
    """
    
    def __init__(self,
                 initial_capital: float = 10000,
                 transaction_cost: float = 0.001,  # 0.1%
                 slippage: float = 0.0005):  # 0.05%
        """
        Initialize backtester
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital in dollars
        transaction_cost : float
            Transaction cost as percentage (0.001 = 0.1%)
        slippage : float
            Slippage as percentage
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        # Trading state
        self.capital = initial_capital
        self.position = 0  # Number of shares held
        self.trades = []
        self.equity_curve = []
        
    def reset(self):
        """Reset backtester state"""
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
    
    def execute_trade(self,
                      date: pd.Timestamp,
                      price: float,
                      signal: int,
                      shares: Optional[int] = None) -> Dict:
        """
        Execute a trade based on signal
        
        Parameters:
        -----------
        date : pd.Timestamp
            Trade date
        price : float
            Execution price
        signal : int
            Trading signal: 1 (buy), -1 (sell), 0 (hold)
        shares : int, optional
            Number of shares to trade. If None, uses full capital
            
        Returns:
        --------
        Dict
            Trade details
        """
        trade = {
            'date': date,
            'signal': signal,
            'price': price,
            'shares': 0,
            'cost': 0,
            'capital_before': self.capital,
            'capital_after': self.capital,
            'position_before': self.position,
            'position_after': self.position
        }
        
        # Apply slippage
        execution_price = price * (1 + self.slippage) if signal == 1 else price * (1 - self.slippage)
        
        if signal == 1 and self.position == 0:  # BUY signal and no position
            # Calculate shares to buy
            if shares is None:
                shares = int(self.capital / (execution_price * (1 + self.transaction_cost)))
            
            if shares > 0:
                total_cost = shares * execution_price * (1 + self.transaction_cost)
                if total_cost <= self.capital:
                    self.capital -= total_cost
                    self.position = shares
                    trade['shares'] = shares
                    trade['cost'] = total_cost
        
        elif signal == -1 and self.position > 0:  # SELL signal and have position
            # Sell all shares
            proceeds = self.position * execution_price * (1 - self.transaction_cost)
            self.capital += proceeds
            trade['shares'] = -self.position
            trade['cost'] = -proceeds
            self.position = 0
        
        trade['capital_after'] = self.capital
        trade['position_after'] = self.position
        
        # Calculate current equity
        current_equity = self.capital + (self.position * price)
        self.equity_curve.append(current_equity)
        
        if trade['shares'] != 0:
            self.trades.append(trade)
        
        return trade
    
    def run_backtest(self,
                     data: pd.DataFrame,
                     predictions: np.ndarray,
                     price_column: str = 'sp500',
                     threshold: float = 0.0) -> Dict:
        """
        Run full backtest on historical data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical data with prices
        predictions : np.ndarray
            Model predictions (same length as data)
        price_column : str
            Column name for prices
        threshold : float
            Threshold for generating buy/sell signals
            - For regression: buy if predicted_change > threshold
            - For classification: threshold on probability
            
        Returns:
        --------
        Dict
            Backtest results and performance metrics
        """
        self.reset()
        
        if len(data) != len(predictions):
            raise ValueError("Data and predictions must have same length")
        
        # Generate signals based on predictions
        signals = self._generate_signals(data[price_column].values, predictions, threshold)
        
        # Execute trades
        for idx, (date, row) in enumerate(data.iterrows()):
            price = row[price_column]
            signal = signals[idx]
            self.execute_trade(date, price, signal)
        
        # Close any open position at end
        if self.position > 0:
            final_price = data[price_column].iloc[-1]
            self.execute_trade(
                data.index[-1],
                final_price,
                signal=-1  # Force sell
            )
        
        # Calculate performance metrics
        results = self._calculate_performance()
        results['trades_log'] = pd.DataFrame(self.trades)
        results['equity_curve'] = np.array(self.equity_curve)
        
        return results
    
    def _generate_signals(self,
                         prices: np.ndarray,
                         predictions: np.ndarray,
                         threshold: float) -> np.ndarray:
        """
        Generate trading signals from predictions
        
        Returns array of: 1 (buy), -1 (sell), 0 (hold)
        """
        signals = np.zeros(len(predictions), dtype=int)
        
        # Calculate predicted price changes
        predicted_changes = predictions - prices
        predicted_returns = predicted_changes / prices
        
        # Track position state: 0 = no position, 1 = holding position
        current_position = 0
        
        for i in range(1, len(signals)):
            # Buy if significant predicted increase AND we don't have a position
            if predicted_returns[i] > threshold and current_position == 0:
                signals[i] = 1
                current_position = 1
            # Sell if significant predicted decrease AND we have a position
            elif predicted_returns[i] < -threshold and current_position == 1:
                signals[i] = -1
                current_position = 0
            else:
                signals[i] = 0
        
        return signals
    
    def _calculate_performance(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {
                'total_return_%': 0.0,
                'num_trades': 0,
                'win_rate_%': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown_%': 0.0
            }
        
        # Calculate returns
        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        # Trade returns
        trade_returns = []
        for i in range(1, len(self.trades)):
            entry_trade = self.trades[i-1]
            exit_trade = self.trades[i]
            if entry_trade['signal'] == 1 and exit_trade['signal'] == -1:
                ret = (exit_trade['price'] - entry_trade['price']) / entry_trade['price']
                trade_returns.append(ret)
        
        # Win rate
        if trade_returns:
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100
        else:
            win_rate = 0.0
        
        # Sharpe ratio
        if trade_returns and np.std(trade_returns) > 0:
            sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max drawdown
        equity_array = np.array(self.equity_curve)
        cumulative_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - cumulative_max) / cumulative_max
        max_dd = np.min(drawdown) * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_equity,
            'total_return_%': total_return,
            'num_trades': len(self.trades),
            'win_rate_%': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown_%': max_dd,
            'avg_trade_return_%': np.mean(trade_returns) * 100 if trade_returns else 0.0,
            'best_trade_%': max(trade_returns) * 100 if trade_returns else 0.0,
            'worst_trade_%': min(trade_returns) * 100 if trade_returns else 0.0
        }


def run_backtest(model_predictions: np.ndarray,
                 test_data: pd.DataFrame,
                 initial_capital: float = 10000,
                 price_column: str = 'sp500') -> Tuple[Dict, pd.DataFrame]:
    """
    Convenience function to run backtest
    
    Parameters:
    -----------
    model_predictions : np.ndarray
        Model predictions
    test_data : pd.DataFrame
        Test dataset with prices
    initial_capital : float
        Starting capital
    price_column : str
        Price column name
        
    Returns:
    --------
    Tuple[Dict, pd.DataFrame]
        Performance metrics and trades log
    """
    backtester = TradingBacktester(initial_capital=initial_capital)
    results = backtester.run_backtest(test_data, model_predictions, price_column)
    
    return results, results['trades_log']


if __name__ == "__main__":
    print("Backtesting module")
    print("Simulates trading strategies with realistic transaction costs")
