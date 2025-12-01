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

# Sidebar
st.sidebar.title("Configuration")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Data Explorer", "Train Models", "Predictions", "Backtesting", "Performance"]
)

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

# ===================== HOME PAGE =====================
if page == "Home":
    st.header("Welcome to the Stock Price Prediction System!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Range", "1990-2024", "34 years")
    with col2:
        st.metric("Total Observations", "8,599", "Daily data")
    with col3:
        st.metric("Features", "100+", "Engineered")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What This System Does")
        st.markdown("""
        - **Predicts** S&P 500 price movements
        - **Identifies** optimal buy/sell points
        - **Evaluates** model performance with financial metrics
        - **Backtests** trading strategies with realistic constraints
        """)
        
        st.subheader("Models Available")
        st.markdown("""
        - **XGBoost** - Gradient boosting for regression
        - **LightGBM** - Fast gradient boosting
        - **LSTM** - Deep learning for time series
        """)
    
    with col2:
        st.subheader("Key Features")
        st.markdown("""
        - **Technical Indicators**: SMA, EMA, MACD, RSI, Bollinger Bands
        - **Lag Features**: Historical prices and returns
        - **Volatility Metrics**: VIX analysis and rolling volatility
        - **Financial Metrics**: Sharpe ratio, max drawdown, win rate
        """)
        
        st.subheader("Important Notes")
        st.info("""
        EDUCATIONAL PURPOSE ONLY: Stock prediction is inherently uncertain. 
        Always consult financial professionals and never invest more than you 
        can afford to lose.
        """)
    
    # Load and display data preview
    st.markdown("---")
    st.subheader("Data Preview")
    df = load_data()
    if df is not None:
        st.dataframe(df.head(100), height=300)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current S&P 500", f"${df['sp500'].iloc[-1]:.2f}")
        with col2:
            st.metric("30-Day Change", f"{df['sp500'].pct_change(30).iloc[-1]*100:.2f}%")
        with col3:
            st.metric("Latest VIX", f"{df['vix'].iloc[-1]:.2f}")
        with col4:
            st.metric("Volume (Latest)", f"{df['sp500_volume'].iloc[-1]/1e6:.1f}M")

# ===================== DATA EXPLORER =====================
elif page == "Data Explorer":
    st.header("Data Explorer")
    
    df = load_data()
    if df is None:
        st.error("Cannot load data!")
        st.stop()
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=df.index.min())
    with col2:
        end_date = st.date_input("End Date", value=df.index.max())
    
    # Filter data
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    filtered_df = df[mask]
    
    # Plot S&P 500
    st.subheader("S&P 500 Historical Price")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(filtered_df.index, filtered_df['sp500'], linewidth=1.5, color='steelblue')
    ax.set_xlabel('Date')
    ax.set_ylabel('S&P 500 Price')
    ax.set_title('S&P 500 Price Over Time')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Multiple features plot
    st.subheader("Feature Comparison")
    features_to_plot = st.multiselect(
        "Select features to plot",
        options=['sp500', 'vix', 'djia', 'hsi', 'sp500_volume'],
        default=['sp500', 'vix']
    )
    
    if features_to_plot:
        fig, ax = plt.subplots(figsize=(14, 6))
        for feature in features_to_plot:
            if feature in filtered_df.columns:
                # Normalize for comparison
                normalized = (filtered_df[feature] - filtered_df[feature].min()) / \
                           (filtered_df[feature].max() - filtered_df[feature].min())
                ax.plot(filtered_df.index, normalized, label=feature, linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Normalized Feature Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Statistics
    st.subheader("Statistical Summary")
    st.dataframe(filtered_df.describe())

# ===================== TRAIN MODELS =====================
elif page == "Train Models":
    st.header("Train Prediction Models")
    
    df_features = load_engineered_features()
    if df_features is None:
        st.error("Please run the feature engineering notebook first!")
        st.code("jupyter notebook notebooks/02_feature_engineering.ipynb")
        st.stop()
    
    st.success(f"Features loaded: {df_features.shape[1]} columns, {df_features.shape[0]} rows")
    
    # Target selection
    st.subheader("1. Select Target Variable")
    target_options = [col for col in df_features.columns if 'target' in col]
    target_col = st.selectbox("Choose prediction target", target_options)
    
    # Model selection
    st.subheader("2. Select Model")
    model_type = st.selectbox("Choose model type", ["XGBoost", "LightGBM"])
    
    # Train/test split configuration
    st.subheader("3. Configure Train/Test Split")
    col1, col2 = st.columns(2)
    with col1:
        train_end = st.date_input("Training end date", value=pd.to_datetime('2018-12-31'))
    with col2:
        val_end = st.date_input("Validation end date", value=pd.to_datetime('2021-12-31'))
    
    # Training button
    if st.button("Train Model", type="primary"):
        with st.spinner(f"Training {model_type} model..."):
            try:
                # Prepare data
                target_columns = [col for col in df_features.columns if 'target' in col]
                feature_cols = [col for col in df_features.columns if col not in target_columns and col != 'sp500']
                
                X = df_features[feature_cols]
                y = df_features[target_col]
                
                # Drop missing
                valid_idx = y.notna()
                X = X[valid_idx]
                y = y[valid_idx]
                
                # Split
                train_data, val_data, test_data = temporal_train_test_split(
                    pd.concat([X, y, df_features.loc[valid_idx, 'sp500']], axis=1),
                    train_end=train_end.strftime('%Y-%m-%d'),
                    val_end=val_end.strftime('%Y-%m-%d')
                )
                
                X_train = train_data[feature_cols]
                y_train = train_data[target_col]
                X_val = val_data[feature_cols]
                y_val = val_data[target_col]
                X_test = test_data[feature_cols]
                y_test = test_data[target_col]
                
                # Train
                if model_type == "XGBoost":
                    model, evals_result = train_xgboost(X_train, y_train, X_val, y_val)
                    predictions = predict_xgboost(model, X_test)
                else:
                    model, evals_result = train_lightgbm(X_train, y_train, X_val, y_val)
                    predictions = predict_lightgbm(model, X_test)
                
                # Evaluate
                metrics = regression_metrics_report(y_test.values, predictions)
                
                st.success("Model trained successfully!")
                
                # Display metrics
                st.subheader("Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{metrics['MAE']:.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                with col3:
                    st.metric("R² Score", f"{metrics['R2']:.4f}")
                with col4:
                    st.metric("Dir. Accuracy", f"{metrics['Directional_Accuracy']:.2f}%")
                
                # Store in session state
                st.session_state['model'] = model
                st.session_state['model_type'] = model_type
                st.session_state['predictions'] = predictions
                st.session_state['y_test'] = y_test
                st.session_state['test_data'] = test_data
                st.session_state['metrics'] = metrics
                
            except Exception as e:
                st.error(f"Error training model: {e}")
                import traceback
                st.code(traceback.format_exc())

# ===================== PREDICTIONS =====================
elif page == "Predictions":
    st.header("Model Predictions")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first!")
        st.info("Go to 'Train Models' page to train a model.")
        st.stop()
    
    st.success(f"Model loaded: {st.session_state['model_type']}")
    
    predictions = st.session_state['predictions']
    y_test = st.session_state['y_test']
    test_data = st.session_state['test_data']
    
    # Predictions vs Actual
    st.subheader("Predictions vs Actual Prices")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_data.index, y_test.values, label='Actual', linewidth=2, alpha=0.8)
    ax.plot(test_data.index, predictions, label='Predicted', linewidth=2, alpha=0.8)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('S&P 500 Price', fontsize=12)
    ax.set_title('Actual vs Predicted Stock Prices', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Prediction errors
    st.subheader("Prediction Errors Over Time")
    errors = y_test.values - predictions
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(test_data.index, errors, color=['red' if e < 0 else 'green' for e in errors], alpha=0.6)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Error (Actual - Predicted)')
    ax.set_title('Prediction Errors')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Scatter plot
    st.subheader("Prediction Scatter Plot")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test.values, predictions, alpha=0.5, s=30)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Price', fontsize=12)
    ax.set_ylabel('Predicted Price', fontsize=12)
    ax.set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ===================== BACKTESTING =====================
elif page == "Backtesting":
    st.header("Trading Strategy Backtesting")
    
    if 'predictions' not in st.session_state:
        st.warning("Please train a model first!")
        st.stop()
    
    # Backtesting configuration
    st.subheader("Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
    with col2:
        transaction_cost = st.number_input("Transaction Cost (%)", value=0.1, step=0.05) / 100
    with col3:
        slippage = st.number_input("Slippage (%)", value=0.05, step=0.01) / 100
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                predictions = st.session_state['predictions']
                test_data = st.session_state['test_data']
                
                # Run backtest
                backtester = TradingBacktester(
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    slippage=slippage
                )
                
                results = backtester.run_backtest(
                    test_data,
                    predictions,
                    price_column='sp500'
                )
                
                st.session_state['backtest_results'] = results
                
                st.success("Backtest completed!")
                
                # Display results
                st.subheader("Backtest Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{results['total_return_%']:.2f}%")
                with col2:
                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{results['max_drawdown_%']:.2f}%")
                with col4:
                    st.metric("Win Rate", f"{results['win_rate_%']:.2f}%")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Capital", f"${results['final_capital']:.2f}")
                with col2:
                    st.metric("Total Trades", results['num_trades'])
                with col3:
                    st.metric("Best Trade", f"{results.get('best_trade_%', 0):.2f}%")
                with col4:
                    st.metric("Worst Trade", f"{results.get('worst_trade_%', 0):.2f}%")
                
                # Equity curve
                if len(results['equity_curve']) > 0:
                    st.subheader("Portfolio Equity Curve")
                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.plot(results['equity_curve'], linewidth=2, color='green')
                    ax.axhline(initial_capital, color='red', linestyle='--', 
                              label=f'Initial Capital (${initial_capital})', linewidth=1.5)
                    ax.fill_between(range(len(results['equity_curve'])), 
                                   initial_capital, results['equity_curve'], 
                                   alpha=0.3, color='green')
                    ax.set_xlabel('Trade Number')
                    ax.set_ylabel('Portfolio Value ($)')
                    ax.set_title('Portfolio Performance Over Time', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Trades log
                if len(results['trades_log']) > 0:
                    st.subheader("Trade History")
                    st.dataframe(results['trades_log'])
                
            except Exception as e:
                st.error(f"Error in backtesting: {e}")

# ===================== PERFORMANCE =====================
elif page == "Performance":
    st.header("Model Performance Analysis")
    
    if 'metrics' not in st.session_state:
        st.warning("Please train a model first!")
        st.stop()
    
    metrics = st.session_state['metrics']
    
    # Metrics overview
    st.subheader("Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Regression Metrics")
        for metric, value in metrics.items():
            if metric != 'Confusion_Matrix':
                st.metric(metric, f"{value:.4f}")
    
    with col2:
        if 'backtest_results' in st.session_state:
            st.markdown("### Financial Metrics")
            br = st.session_state['backtest_results']
            st.metric("Total Return", f"{br['total_return_%']:.2f}%")
            st.metric("Sharpe Ratio", f"{br['sharpe_ratio']:.2f}")
            st.metric("Max Drawdown", f"{br['max_drawdown_%']:.2f}%")
            st.metric("Win Rate", f"{br['win_rate_%']:.2f}%")
            st.metric("Profit Factor", f"{br.get('profit_factor', 0):.2f}")
    
    # Comparison chart
    st.subheader("Metrics Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = [k for k in metrics.keys() if k != 'Confusion_Matrix']
    metric_values = [metrics[k] for k in metric_names]
    ax.barh(metric_names, metric_values, color='steelblue')
    ax.set_xlabel('Value')
    ax.set_title('Model Performance Metrics', fontweight='bold')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Stock Price Prediction System | Built with Streamlit, XGBoost, LightGBM, LSTM</p>
    <p><i>WARNING: For educational purposes only. Not financial advice.</i></p>
</div>
""", unsafe_allow_html=True)
