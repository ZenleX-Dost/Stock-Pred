# Stock Price Prediction System

A lightweight machine learning dashboard for predicting S&P 500 price movements using 34 years of historical market data (1990-2024).

## Project Overview

This Streamlit application uses gradient boosting models (XGBoost and LightGBM) to predict S&P 500 price movements and identify optimal buy/sell points. The system combines technical indicators, market data, and financial metrics for production-grade predictions.

## Dataset

**34-Year Daily Stock Data (1990-2024)**
- **Size**: 8,599 observations
- **Features**: 100+ engineered features
- **Temporal Range**: 1990-01-03 to 2024

### Features:
- **Market Indices**: S&P 500, DJIA, Hang Seng Index
- **Volatility**: VIX (Fear Index)
- **Economic Indicators**: Treasury Yield, Unemployment
- **Uncertainty Metrics**: Economic Policy Uncertainty (EPU), Geopolitical Risk
- **Volume Data**: Trading volumes

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

## Quick Start

### Run the Streamlit Application
```bash
streamlit run app.py
```

Then open your browser to http://localhost:8501

### Application Pages

1. **Home** - System overview, data preview, key statistics
2. **Data Explorer** - Interactive historical price visualization and trend analysis
3. **Train Models** - Train XGBoost or LightGBM models on custom date ranges
4. **Predictions** - Model predictions vs actual prices with error analysis
5. **Backtesting** - Trading strategy simulation with transaction costs
6. **Performance** - Regression and financial metrics comparison

## Project Structure

```
Stock-Pred/
├── app.py                      # Streamlit application
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── STREAMLIT_GUIDE.md          # Usage guide
├── stock_data.csv              # Raw dataset
├── data/
│   └── processed/
│       └── features_engineered.csv  # Engineered features
├── models/
│   ├── xgboost_next_day.json   # XGBoost model
│   └── lightgbm_next_day.txt   # LightGBM model
└── src/
    ├── preprocessing.py        # Data preprocessing
    ├── models/
    │   ├── xgboost_model.py
    │   └── lightgbm_model.py
    ├── evaluation/
    │   └── metrics.py
    └── backtesting/
        └── strategy.py
```

## Production Models

### XGBoost
- Fast gradient boosting with strong regularization
- Handles non-linear relationships well
- Production-grade reliability

### LightGBM
- Ultra-fast gradient boosting
- Lower memory footprint
- Excellent for large datasets

Both models are pre-trained and ready for production use.

## Key Metrics

The application provides:
- **Regression Metrics**: MAE, RMSE, R[SQUARED] Score, Directional Accuracy
- **Financial Metrics**: Sharpe Ratio, Max Drawdown, Win Rate, Total Return
- **Backtesting**: Portfolio performance with realistic transaction costs and slippage

## Important Considerations

- **No Lookahead Bias**: All features use only historical data
- **Temporal Validation**: Proper time series splits (train/validation/test)
- **Market Efficiency**: Results acknowledge inherent market uncertainty
- **Transaction Costs**: Backtesting includes 0.1% commission per trade
- **Educational Purpose**: This system is for learning and research only

## Dependencies

- Python 3.8+
- pandas, numpy, scikit-learn
- XGBoost, LightGBM
- matplotlib, seaborn
- statsmodels, scipy
- streamlit

See `requirements.txt` for exact versions.

## Disclaimer

EDUCATIONAL PURPOSE ONLY: This project is for learning and research purposes. Stock market predictions are inherently uncertain. Do not use this system as the sole basis for investment decisions. Always consult financial professionals and conduct your own research before investing.

## License

MIT License

## References

- Historical financial data from Yahoo Finance, CBOE, Federal Reserve
- Methodology based on modern time series forecasting and financial ML best practices
