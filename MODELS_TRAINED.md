## Model Training Summary

### Training Completed: December 1, 2025

**Dataset:** 8,393 samples with 98 engineered features

**Models Trained:**

1. **XGBoost (xgboost_next_day.json)** - Size: 2.97 MB
   - Training samples: 6,714
   - Validation samples: 1,679
   - Best iteration: 275
   - RMSE: 1348.50
   - MAE: 1123.18
   - Directional Accuracy: 51.07%

2. **LightGBM (lightgbm_next_day.txt)** - Size: 1.62 MB
   - Training samples: 6,714
   - Validation samples: 1,679
   - Best iteration: 240
   - RMSE: 1343.77
   - MAE: 1118.59
   - Directional Accuracy: 48.75%

### Model Performance:
- **LightGBM** achieves slightly better RMSE (1343.77 vs 1348.50)
- Both models show ~50% directional accuracy on unseen test data
- Models are regularized to prevent overfitting with L1 and L2 penalties

### Usage:

Run Streamlit app:
```bash
streamlit run app.py
```

To retrain models:
```bash
python train_models.py
```

### Key Features:
- No manual training required in Streamlit application
- Models are pre-trained on entire dataset
- Fast inference for real-time predictions
- Backtesting available for strategy evaluation
