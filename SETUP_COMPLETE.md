## Stock-Pred Streamlit Application - Setup Complete

### Application is Live!

Access the application at: `http://localhost:8502`

### Pre-trained Models Ready

Both models have been trained on the entire dataset (8,393 samples):

1. **XGBoost** (`models/xgboost_next_day.json`)
   - Performance: RMSE 1348.50, MAE 1123.18
   - Size: 2.97 MB
   - 275 iterations

2. **LightGBM** (`models/lightgbm_next_day.txt`)
   - Performance: RMSE 1343.77, MAE 1118.59
   - Size: 1.62 MB
   - 240 iterations

### Application Features

1. **Home Page**
   - Overview of the system
   - Data preview (8,599 daily observations from 1990-2024)
   - Key metrics (S&P 500, VIX, Volume)

2. **Data Explorer**
   - Interactive historical price visualization
   - Multi-feature comparison with normalization
   - Statistical summaries

3. **Train Models** (Pre-trained Evaluation)
   - Load pre-trained XGBoost/LightGBM models
   - Select custom test date range
   - Evaluate model performance
   - View regression metrics

4. **Predictions**
   - Actual vs Predicted price comparison
   - Prediction error analysis
   - Scatter plot for accuracy visualization

5. **Backtesting**
   - Trading strategy simulation
   - Configurable initial capital, transaction costs, slippage
   - Portfolio equity curve visualization
   - Trade history logging

6. **Performance**
   - Comprehensive metrics dashboard
   - Regression statistics (MAE, RMSE, R²)
   - Financial metrics (Sharpe ratio, max drawdown, win rate)

### Running the Application

Start the app:
```bash
streamlit run app.py
```

Retrain models (if needed):
```bash
python train_models.py
```

### Project Structure

```
Stock-Pred/
├── app.py                          # Streamlit application
├── train_models.py                 # Model training script
├── requirements.txt                # Dependencies
├── stock_data.csv                  # Raw data (1990-2024)
├── models/
│   ├── xgboost_next_day.json      # Pre-trained XGBoost
│   └── lightgbm_next_day.txt      # Pre-trained LightGBM
├── data/
│   └── processed/
│       └── features_engineered.csv # 100+ engineered features
└── src/
    ├── preprocessing.py
    ├── models/
    │   ├── xgboost_model.py
    │   └── lightgbm_model.py
    ├── evaluation/metrics.py
    └── backtesting/strategy.py
```

### Notes

- Models are cached in memory for fast inference
- No training required in the app itself
- Features include technical indicators, lag features, volatility metrics
- All predictions are made in real-time when evaluating test periods
- App supports backtesting with realistic trading constraints

### Next Steps

1. Open browser to `http://localhost:8502`
2. Start with "Data Explorer" to understand the dataset
3. Go to "Train Models" to evaluate pre-trained models on different date ranges
4. Use "Backtesting" to simulate trading strategies
5. Review "Performance" metrics for detailed analysis
