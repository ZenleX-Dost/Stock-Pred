"""
Stock Price Prediction Dashboard - Streamlit Web Application

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from preprocessing import temporal_train_test_split, scale_features
from models.xgboost_model import train_xgboost, predict_xgboost, load_model as load_xgb
from models.lightgbm_model import train_lightgbm, predict_lightgbm, load_model as load_lgb
from evaluation.metrics import regression_metrics_report, financial_metrics_report
from backtesting.strategy import TradingBacktester

# Cache model loading
@st.cache_resource
def load_pretrained_models():
    """Load pre-trained XGBoost and LightGBM models"""
    try:
        xgb_model = load_xgb('models/xgboost_next_day.json')
        lgb_model = load_lgb('models/lightgbm_next_day.txt')
        return xgb_model, lgb_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="[CHART]",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Stock Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown("**Machine Learning-Powered S&P 500 Predictions (1990-2024)**")
st.markdown("---")

# Cache data loading
@st.cache_data
def load_data():
    """Load stock data"""
    try:
        df = pd.read_csv('stock_data.csv', parse_dates=['dt'])
        df.set_index('dt', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_engineered_features():
    """Load engineered features"""
    try:
        df = pd.read_csv('data/processed/features_engineered.csv', parse_dates=['dt'])
        df.set_index('dt', inplace=True)
        return df
    except Exception as e:
        st.warning("Engineered features not found. Please run feature engineering notebook first.")
        return None

# Sidebar
st.sidebar.title("Configuration")
model_type = st.sidebar.selectbox("Select Model", ["XGBoost", "LightGBM"])

# Load data
df_features = load_engineered_features()
df_raw = load_data()

if df_features is None or df_raw is None:
    st.stop()

# Date Selection
min_date = df_features.index.min().date()
max_date = df_features.index.max().date()

st.sidebar.markdown("---")
prediction_mode = st.sidebar.radio("Prediction Mode", ["Historical Data", "Future Scenario"])

if prediction_mode == "Historical Data":
    selected_date = st.sidebar.date_input(
        "Select Date for Prediction",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
else:
    selected_date = st.sidebar.date_input(
        "Select Future Date",
        value=pd.Timestamp.now().date()
    )

# Load Models
xgb_model, lgb_model = load_pretrained_models()
if xgb_model is None or lgb_model is None:
    st.error("Models not found. Please run train_models.py first.")
    st.stop()

model = xgb_model if model_type == "XGBoost" else lgb_model

# Main Content
st.title(f"S&P 500 Prediction Dashboard")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Prediction")
    
    # Handle date selection
    check_date = pd.Timestamp(selected_date)
    
    if prediction_mode == "Historical Data":
        # Find nearest available date if selected date is not in index
        if check_date not in df_features.index:
            # Find nearest date
            nearest_idx = df_features.index.get_indexer([check_date], method='nearest')[0]
            actual_date = df_features.index[nearest_idx]
            st.warning(f"Data not available for {selected_date}. Using nearest date: {actual_date.date()}")
            check_date = actual_date
        
        # Get features for the date
        row = df_features.loc[[check_date]]
        
    else:
        # Future Scenario Mode
        st.info("Scenario Mode: Using latest known market data as baseline. Adjust key indicators below.")
        
        # Get latest known data as baseline
        latest_date = df_features.index.max()
        baseline_row = df_features.loc[[latest_date]].copy()
        
        # Allow user to adjust key features
        current_price = st.number_input("Current S&P 500 Price", value=float(baseline_row['sp500'].values[0]))
        current_vix = st.number_input("Current VIX", value=float(baseline_row['vix'].values[0]))
        
        # Update baseline row with user inputs
        baseline_row['sp500'] = current_price
        baseline_row['vix'] = current_vix
        
        # Use this modified row for prediction
        row = baseline_row
        check_date = pd.Timestamp(selected_date) # Use user's future date for display

    # Prepare features
    target_columns = [col for col in df_features.columns if 'target' in col]
    feature_cols = [col for col in df_features.columns if col not in target_columns and col != 'sp500']
    X_input = row[feature_cols]
    
    # Predict
    if model_type == "XGBoost":
        prediction = predict_xgboost(model, X_input)[0]
    else:
        prediction = predict_lightgbm(model, X_input)[0]
    
    # Display Prediction
    st.metric("Predicted Price (Next Day)", f"${prediction:.2f}")
    
    if prediction_mode == "Historical Data":
        # If we have actual target for this date (meaning we know what happened next day)
        # Note: target_next_day is the price of the NEXT day.
        # If we want to compare with actual price of the NEXT day:
        if len(target_columns) > 0:
            target_col = target_columns[0]
            if not pd.isna(row[target_col].values[0]):
                actual_next_day = row[target_col].values[0]
                diff = actual_next_day - prediction
                pct_error = (diff / actual_next_day) * 100
                
                st.metric("Actual Next Day Price", f"${actual_next_day:.2f}")
                st.metric("Prediction Error", f"${diff:.2f} ({pct_error:.2f}%)")
                
                if abs(pct_error) < 1.0:
                    st.success("High Accuracy Prediction!")
                elif abs(pct_error) < 3.0:
                    st.info("Moderate Accuracy")
                else:
                    st.warning("Low Accuracy")
            else:
                st.info("Actual next day price not yet available in dataset.")
    else:
        # Future mode - show implied change
        current_p = row['sp500'].values[0]
        change = prediction - current_p
        pct = (change / current_p) * 100
        st.metric("Implied Change", f"{change:.2f} ({pct:.2f}%)")

with col2:
    st.subheader("Market Data Visualization")
    
    if prediction_mode == "Historical Data":
        # Filter data for chart (show +/- 180 days around selected date)
        window_days = 180
        start_plot = check_date - pd.Timedelta(days=window_days)
        end_plot = check_date + pd.Timedelta(days=window_days)
        
        mask = (df_raw.index >= start_plot) & (df_raw.index <= end_plot)
        plot_data = df_raw[mask]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(plot_data.index, plot_data['sp500'], label='S&P 500', linewidth=2)
        
        # Highlight the selected date (which is the date OF prediction, i.e. today)
        current_price = row['sp500'].values[0] if 'sp500' in row else None
        if current_price:
            ax.scatter([check_date], [current_price], color='black', s=100, zorder=5, label='Selected Date')
        
        # Highlight the predicted next day price
        # We plot it at check_date + 1 day (approx)
        next_day_date = check_date + pd.Timedelta(days=1)
        ax.scatter([next_day_date], [prediction], color='red', s=100, marker='*', zorder=5, label='Prediction (Next Day)')
        
        ax.set_title(f"S&P 500 Price Context ({start_plot.date()} - {end_plot.date()})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    else:
        # Future mode visualization
        # Show recent history + projected point
        recent_days = 90
        plot_data = df_raw.tail(recent_days)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(plot_data.index, plot_data['sp500'], label='Historical S&P 500', linewidth=2)
        
        # Plot the "Current" scenario point (user input)
        scenario_date = check_date
        scenario_price = row['sp500'].values[0]
        ax.scatter([scenario_date], [scenario_price], color='orange', s=100, zorder=5, label='Scenario Input (Current)')
        
        # Plot the prediction (Next Day)
        pred_date = scenario_date + pd.Timedelta(days=1)
        ax.scatter([pred_date], [prediction], color='red', s=100, marker='*', zorder=5, label='Prediction (Next Day)')
        
        # Connect the lines
        last_hist_date = plot_data.index[-1]
        last_hist_price = plot_data['sp500'].iloc[-1]
        
        # Draw dashed line from history to scenario point
        ax.plot([last_hist_date, scenario_date], [last_hist_price, scenario_price], 'k--', alpha=0.5)
        # Draw dashed line from scenario to prediction
        ax.plot([scenario_date, pred_date], [scenario_price, prediction], 'r--', alpha=0.5)
        
        ax.set_title(f"Future Scenario Projection")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

st.markdown("---")
st.subheader("Feature Insights")
with st.expander("View Input Features"):
    st.dataframe(row)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Stock Price Prediction System | Built with Streamlit, XGBoost, LightGBM</p>
</div>
""", unsafe_allow_html=True)
