#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETH/BTC Pair Trading Strategy using GARCH and LSTM Models
This script implements a pair trading strategy using GARCH and LSTM models
to trade the ETH/BTC pair, including volatility forecasting and
altcoin season filtering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import shap
import tensorflow as tf
from arch import arch_model
from datetime import datetime
from dotenv import load_dotenv
from statsmodels.stats.diagnostic import acorr_ljungbox
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, MultiHeadAttention, Layer, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

def fetch_crypto_data(start_date, end_date):
    """Fetch historical crypto data from Alpaca."""
    client = CryptoHistoricalDataClient(
        api_key=os.getenv("APCA_API_KEY_ID"),
        secret_key=os.getenv("APCA_API_SECRET_KEY")
    )
    
    request_params = CryptoBarsRequest(
        symbol_or_symbols=["BTC/USDT", "ETH/USDT"],
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )
    
    bars = client.get_crypto_bars(request_params)
    return bars.df

def preprocess_data(df):
    """Preprocess the crypto data for analysis."""
    # Reset index and rename timestamp column
    df = df.reset_index()
    datetime_col = 'timestamp' if 'timestamp' in df.columns else 'index'
    df = df.rename(columns={datetime_col: 'datetime'})
    
    # Sort and pivot data
    df = df.sort_values(by=['symbol', 'datetime'])
    price_df = df.pivot(index='datetime', columns='symbol', values='close')
    price_df.columns = ['btc_close', 'eth_close']
    price_df.dropna(inplace=True)
    
    # Calculate returns and scale prices
    price_df['btc_return'] = np.log(price_df['btc_close'] / price_df['btc_close'].shift(1))
    price_df['eth_return'] = np.log(price_df['eth_close'] / price_df['eth_close'].shift(1))
    price_df.dropna(inplace=True)
    
    scaler = MinMaxScaler()
    price_df[['btc_close_scaled', 'eth_close_scaled']] = scaler.fit_transform(
        price_df[['btc_close', 'eth_close']]
    )
    
    return price_df

def plot_price_data(price_df):
    """Plot BTC and ETH closing prices."""
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('BTC Closing Price Over Time', 
                                     'ETH Closing Price Over Time'))
    
    fig.add_trace(
        go.Scatter(x=price_df.index, y=price_df['btc_close'], 
                  mode='lines', name='BTC Close', 
                  line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=price_df.index, y=price_df['eth_close'], 
                  mode='lines', name='ETH Close', 
                  line=dict(color='green')),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Daily Closing Prices Over Time',
        xaxis_title='Date',
        yaxis_title='Closing Price (USDT)',
        template='plotly',
        height=600
    )
    
    fig.update_yaxes(title_text='Closing Price (USDT)', row=1, col=1)
    fig.update_yaxes(title_text='Closing Price (USDT)', row=2, col=1)
    fig.show()

def plot_returns(price_df):
    """Plot BTC and ETH log returns."""
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('BTC Log Return Over Time', 
                                     'ETH Log Return Over Time'))
    
    fig.add_trace(
        go.Scatter(x=price_df.index, y=price_df['btc_return'], 
                  mode='lines', name='BTC Log Return', 
                  line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=price_df.index, y=price_df['eth_return'], 
                  mode='lines', name='ETH Log Return', 
                  line=dict(color='green')),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Log Returns Over Time',
        xaxis_title='Date',
        yaxis_title='Log Return',
        template='plotly',
        height=600
    )
    
    fig.update_yaxes(title_text='Log Return', row=1, col=1)
    fig.update_yaxes(title_text='Log Return', row=2, col=1)
    fig.show()

def egarch_analysis(returns, symbol, rolling_window=21):
    """Perform EGARCH analysis on returns data."""
    print(f"\n=== EGARCH(1,1) Model for {symbol} ===")
    
    # Fit EGARCH model
    am = arch_model(returns * 100, vol='EGARCH', p=1, q=1, 
                   dist='normal', mean='Zero')
    res = am.fit(update_freq=5, disp='off')
    print(res.summary())
    
    # Extract metrics
    print(f"AIC: {res.aic:.4f}, BIC: {res.bic:.4f}")
    
    # Ljung-Box test
    lb_test = acorr_ljungbox(res.std_resid ** 2, lags=[10], return_df=True)
    print("\nLjung-Box test for ARCH effects (p-value):")
    print(lb_test)
    
    # Calculate volatilities
    realized_vol = returns.rolling(window=rolling_window).std() * np.sqrt(252) * 100
    cond_vol = res.conditional_volatility * np.sqrt(252)
    
    # Plot volatilities
    plt.figure(figsize=(12,5))
    plt.plot(realized_vol.index, realized_vol, 
             label='Realized Volatility (Rolling STD)')
    plt.plot(cond_vol.index, cond_vol, 
             label='EGARCH Conditional Volatility', alpha=0.75)
    plt.title(f'Volatility Comparison for {symbol}')
    plt.ylabel('Volatility (%)')
    plt.legend()
    plt.show()
    
    return res

def create_sequences(df, seq_len=30):
    """Create sequences for LSTM model."""
    X, y_signal, y_vol = [], [], []
    for i in range(len(df) - seq_len):
        window = df[['spread', 'btc_vol', 'eth_vol']].iloc[i:i+seq_len].values
        fut = df['spread'].iloc[i+seq_len]
        mu = df['spread'].iloc[i:i+seq_len].mean()
        sigma = df['spread'].iloc[i:i+seq_len].std()
        
        # Classification target
        if fut < mu - sigma:
            sig = [1, 0, 0]  # Long
        elif fut > mu + sigma:
            sig = [0, 1, 0]  # Short
        else:
            sig = [0, 0, 1]  # Neutral
            
        X.append(window)
        y_signal.append(sig)
        y_vol.append(fut)
        
    return np.array(X), np.array(y_signal), np.array(y_vol)

def build_lstm_attention_model(seq_len, n_features, n_heads=4):
    """Build LSTM model with attention mechanism."""
    inp = Input((seq_len, n_features))
    x = tf.keras.layers.Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Dropout(0.2)(x)
    
    # Multi-head attention
    attn = MultiHeadAttention(num_heads=n_heads, key_dim=64)(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn)
    
    # Temporal attention
    scores = Dense(1, activation='tanh')(x)
    weights = tf.keras.layers.Softmax(axis=1)(scores)
    context = Lambda(lambda t: tf.reduce_sum(t[0]*t[1], axis=1))([x, weights])
    
    # Outputs
    signal_out = Dense(3, activation='softmax', name='signal')(context)
    vol_out = Dense(1, activation='linear', name='vol')(context)
    
    model = Model(inp, [signal_out, vol_out])
    model.compile(
        optimizer=Adam(1e-3),
        loss={'signal': 'categorical_crossentropy', 'vol': 'mse'},
        metrics={'signal': 'accuracy', 'vol': 'mse'}
    )
    
    return model

def main():
    """Main execution function."""
    # Load environment variables
    load_dotenv()
    
    # Set date range
    start_date = datetime(2021, 1, 1)
    end_date = datetime.now()
    
    # Fetch and preprocess data
    print("Fetching crypto data...")
    df = fetch_crypto_data(start_date, end_date)
    price_df = preprocess_data(df)
    
    # Plot data
    print("\nPlotting price data...")
    plot_price_data(price_df)
    plot_returns(price_df)
    
    # Train-test split
    split_index = int(len(price_df) * 0.8)
    train_df = price_df.iloc[:split_index]
    test_df = price_df.iloc[split_index:]
    
    print("\nData split:")
    print(f"Train Data Range: {train_df.index.min().date()} to {train_df.index.max().date()}")
    print(f"Test Data Range: {test_df.index.min().date()} to {test_df.index.max().date()}")
    
    # EGARCH analysis
    print("\nPerforming EGARCH analysis...")
    btc_train_returns = train_df['btc_return'].dropna()
    eth_train_returns = train_df['eth_return'].dropna()
    
    btc_res_train = egarch_analysis(btc_train_returns, 'BTC/USDT (Train Set)')
    eth_res_train = egarch_analysis(eth_train_returns, 'ETH/USDT (Train Set)')
    
    # Prepare data for LSTM
    print("\nPreparing data for LSTM model...")
    price_df['spread'] = price_df['eth_close_scaled'] - price_df['btc_close_scaled']
    price_df['btc_vol'] = np.nan
    price_df['eth_vol'] = np.nan
    
    # Add volatility forecasts
    price_df.loc[btc_res_train.conditional_volatility.index, 'btc_vol'] = \
        btc_res_train.conditional_volatility / 100
    price_df.loc[eth_res_train.conditional_volatility.index, 'eth_vol'] = \
        eth_res_train.conditional_volatility / 100
    
    price_df.dropna(subset=['spread', 'btc_vol', 'eth_vol'], inplace=True)
    
    # Create sequences
    seq_len = 30
    X, y_signal, y_vol = create_sequences(price_df, seq_len)
    
    # Train-test split for LSTM
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_signal[:train_size], y_signal[train_size:]
    vol_train, vol_test = y_vol[:train_size], y_vol[train_size:]
    
    # Build and train LSTM model
    print("\nBuilding and training LSTM model...")
    model = build_lstm_attention_model(seq_len, n_features=3)
    model.summary()
    
    # Training callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_signal_accuracy',
        patience=5,
        mode='max',
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train,
        {'signal': y_train, 'vol': vol_train},
        validation_data=(X_test, {'signal': y_test, 'vol': vol_test}),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping]
    )
    
    # Plot training history
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['signal_accuracy'], label='Train Acc')
    plt.plot(history.history['val_signal_accuracy'], label='Val Acc')
    plt.title('Signal Accuracy')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['vol_mse'], label='Train MSE')
    plt.plot(history.history['val_vol_mse'], label='Val MSE')
    plt.title('Volatility Forecast MSE')
    plt.legend()
    plt.show()
    
    # Backtesting
    print("\nPerforming backtesting...")
    y_pred_probs, y_vol_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred_probs, axis=1)
    
    signal_map = {0: 1, 1: -1, 2: 0}  # Long, Short, Neutral
    predicted_signals = np.array([signal_map[c] for c in y_pred_class])
    
    # Calculate returns
    spread_test = y_vol[train_size:]
    spread_returns = np.diff(spread_test, prepend=spread_test[0])
    strategy_returns = predicted_signals * spread_returns
    
    # Add transaction costs
    transaction_cost = 0.001
    signal_change = np.abs(np.diff(predicted_signals, prepend=0))
    costs = signal_change * transaction_cost
    net_returns = strategy_returns - costs
    cumulative_returns = np.cumprod(1 + net_returns)
    
    # Calculate performance metrics
    total_return = cumulative_returns[-1]
    sharpe_ratio = np.mean(net_returns) / np.std(net_returns) * np.sqrt(252)
    max_drawdown = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))
    
    print("\nStrategy Performance:")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
    
    # Plot performance
    plt.figure(figsize=(10,5))
    plt.plot(cumulative_returns, label='Strategy Cumulative Return')
    plt.title('Backtest Equity Curve')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main() 
