# Running the Streamlit Dashboard

## Quick Start

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Pages Available

1. **🏠 Home** - Overview and data preview
2. **📊 Data Explorer** - Interactive data visualization
3. **🤖 Train Models** - Train XGBoost/LightGBM models
4. **📈 Predictions** - View model predictions vs actual
5. **💰 Backtesting** - Test trading strategies
6. **📉 Performance** - Comprehensive metrics analysis

## Prerequisites

Make sure you've:
1. Installed dependencies: `pip install -r requirements.txt`
2. Run feature engineering: `jupyter notebook notebooks/02_feature_engineering.ipynb`

## Features

- **Interactive Charts** - Zoom, pan, and explore data
- **Real-time Training** - Train models directly in the browser
- **Backtesting** - Simulate trading with configurable parameters
- **Performance Metrics** - Sharpe ratio, max drawdown, win rate

## Configuration

Customize in the sidebar:
- Training/validation split dates
- Initial capital for backtesting
- Transaction costs and slippage
- Model hyperparameters

Enjoy! 📈
