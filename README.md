# Stock Price Prediction System

A comprehensive machine learning system for predicting stock price movements using 34 years of historical market data (1990-2024).

## Project Overview

This project implements multiple ML/DL models to predict S&P 500 price movements and identify optimal buy/sell points. The system combines traditional machine learning algorithms (XGBoost, LightGBM, Random Forest) with deep learning models (LSTM, GRU) to provide robust predictions with confidence scores.

## Dataset

**34-Year Daily Stock Data (1990-2024)**
- **Size**: 8,599 observations
- **Features**: 13 columns including market indices, volatility, economic indicators
- **Temporal Range**: 1990-01-03 to 2024-latest

### Features:
- **Market Indices**: S&P 500, DJIA, Hang Seng Index
- **Volatility**: VIX (Fear Index)
- **Economic Indicators**: ADS Business Conditions, US 3-Month Treasury Yield, Unemployment
- **Uncertainty Metrics**: Economic Policy Uncertainty (EPU), Geopolitical Risk (GPRD)
- **Volume Data**: Trading volumes for major indices

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Stock-Pred.git
cd Stock-Pred
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Project Structure

```
Stock-Pred/
├── stock_data.csv              # Raw dataset
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_interpretation.ipynb
├── src/
│   ├── preprocessing.py        # Data preprocessing utilities
│   ├── models/
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── lstm_model.py
│   │   ├── gru_model.py
│   │   └── ensemble.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── backtesting/
│       └── strategy.py
├── data/
│   └── processed/             # Engineered features
├── models/                    # Saved model files
├── results/                   # Evaluation results
├── config/                    # Configuration files
└── requirements.txt
```

## Usage

### 1. Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_eda.ipynb
```
Run all cells to perform comprehensive data exploration including:
- Data quality assessment
- Temporal trend analysis
- Statistical profiling
- Feature correlation analysis

### 2. Feature Engineering
```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```
Creates technical indicators, lag features, and rolling statistics.

### 3. Train Models
```python
# Example: Train XGBoost model
from src.models.xgboost_model import train_xgboost
model = train_xgboost(X_train, y_train, X_val, y_val)
```

### 4. Backtesting
```python
from src.backtesting.strategy import run_backtest
results = run_backtest(model, test_data, initial_capital=10000)
```

## Key Results

*Results will be updated after model training*

### Model Performance Metrics
- Directional Accuracy: TBD
- Sharpe Ratio: TBD
- Maximum Drawdown: TBD

## Methodology

### Data Preprocessing
- Temporal train/test split (NO random shuffling to prevent lookahead bias)
- Feature scaling using StandardScaler/MinMaxScaler
- Handling missing values with forward fill
- Outlier treatment using winsorization

### Models Implemented
1. **Baseline**: Naive forecast, Moving Average, Linear Regression
2. **Traditional ML**: XGBoost, LightGBM, Random Forest
3. **Deep Learning**: LSTM, GRU, Bidirectional LSTM
4. **Ensemble**: Weighted averaging and stacking

### Evaluation
- **Regression Metrics**: MAE, RMSE, R²
- **Classification Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Financial Metrics**: Sharpe Ratio, Max Drawdown, Win Rate

## Critical Considerations

- **No Lookahead Bias**: All features use only historical data
- **Temporal Validation**: Proper time series split (1990-2018 train / 2019-2021 val / 2022-2024 test)
- **Market Efficiency**: Results acknowledge the Efficient Market Hypothesis
- **Transaction Costs**: Backtesting includes 0.1% commission per trade
- **Risk Management**: Predictions should be combined with fundamental analysis

## Dependencies

- Python 3.8+
- pandas, numpy
- scikit-learn
- XGBoost, LightGBM
- TensorFlow/Keras
- matplotlib, seaborn
- statsmodels, scipy

See `requirements.txt` for complete list.

## License

MIT License

## Disclaimer

This project is for educational and research purposes only. Stock market predictions are inherently uncertain. Do not use this system as the sole basis for investment decisions. Always consult financial professionals and conduct your own research.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## References

- Dataset: Historical financial data from Yahoo Finance, CBOE, Federal Reserve
- Methodology: Based on modern time series forecasting and financial ML best practices

## Contact

For questions or collaboration: [your-email@example.com]
