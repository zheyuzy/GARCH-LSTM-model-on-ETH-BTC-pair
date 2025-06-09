# ETH/BTC Pair Trading Strategy with GARCH-LSTM

## Introduction

This project implements a hybrid quantitative trading strategy for the ETH/BTC cryptocurrency pair, combining traditional econometric models (EGARCH) with modern deep learning (LSTM with attention). The goal is to exploit mean-reverting behavior in the ETH/BTC spread, while accounting for volatility clustering and regime changes.

Pair trading is a market-neutral strategy that seeks to profit from the relative price movement between two correlated assets. ETH and BTC are the two largest cryptocurrencies, making their relationship a rich source for statistical arbitrage.

## How It Works

1. **Data Collection:**  
   Historical price data for BTC/USDT and ETH/USDT is fetched from the Alpaca API.
2. **Preprocessing:**  
   - Data is cleaned, aligned, and log returns are calculated.  
   - Prices are normalized for model input.
3. **Volatility Modeling (EGARCH):**  
   - EGARCH(1,1) models are fitted to both BTC and ETH returns.  
   - Conditional volatility forecasts are generated and used as features.
4. **Sequence Creation:**  
   - The spread between ETH and BTC is calculated and used to create time series sequences for the LSTM.
5. **LSTM with Attention:**  
   - A bidirectional LSTM with multi-head attention predicts both the next spread and a trading signal (long, short, neutral).  
   - The model is trained using both classification and regression objectives.
6. **Backtesting:**  
   - The model's signals are used to simulate trades, including transaction costs.  
   - Performance metrics such as Sharpe ratio, total return, and drawdown are calculated.
7. **Visualization:**  
   - The script generates interactive plots for prices, returns, volatility, and strategy performance.

## Features

- **Data Collection**: Automated historical data retrieval from Alpaca API for BTC/USDT and ETH/USDT pairs
- **Volatility Modeling**: EGARCH(1,1) implementation for volatility forecasting
- **Machine Learning**: LSTM model with multi-head attention mechanism for signal generation
- **Risk Management**: Transaction cost consideration and position sizing
- **Performance Analysis**: Comprehensive backtesting with key metrics (Sharpe ratio, drawdown, etc.)
- **Visualization**: Interactive plots for prices, returns, volatility, and strategy performance

## Model Details

- **EGARCH (Exponential GARCH):**  
  Captures volatility clustering and leverage effects, which are common in crypto markets. EGARCH is chosen for its ability to model asymmetric volatility responses.
- **LSTM with Attention:**  
  LSTM networks are well-suited for time series forecasting. The attention mechanism allows the model to focus on the most relevant time steps, improving signal quality.

## Customization

- **Date Range:**  
  Change `start_date` and `end_date` in the script to adjust the backtest period.
- **Sequence Length:**  
  Modify `seq_len` in the script to change the lookback window for the LSTM.
- **Model Hyperparameters:**  
  Adjust LSTM units, attention heads, batch size, and epochs as needed.

## Example Output

After running the script, you will see:
- Plots of BTC and ETH prices and returns
- Volatility comparison (realized vs. EGARCH)
- Training curves for the LSTM
- Backtest equity curve
- Performance metrics (Sharpe ratio, total return, max drawdown)

## Troubleshooting & FAQ

- **Q: I get an API error from Alpaca.**  
  A: Double-check your API keys in the `.env` file and ensure your account has access to crypto data.
- **Q: ModuleNotFoundError for a package.**  
  A: Run `pip install -r requirements.txt` to install all dependencies.
- **Q: The script is slow or uses a lot of memory.**  
  A: Try reducing the date range or LSTM sequence length.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`
- Alpaca API credentials

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ethbtc.git
cd ethbtc
```
2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create a `.env` file with your Alpaca API credentials:
```
APCA_API_KEY_ID=your_api_key_here
APCA_API_SECRET_KEY=your_secret_key_here
```

## Usage

Run the main script:
```bash
python eth_btc_pair_trading.py
```

The script will:
1. Fetch historical data
2. Preprocess and analyze the data
3. Train the GARCH and LSTM models
4. Perform backtesting
5. Display performance metrics and visualizations

## Project Structure

```
ethbtc/
├── eth_btc_pair_trading.py  # Main strategy implementation
├── requirements.txt         # Project dependencies
├── .env                    # API credentials (not tracked in git)
└── README.md                # This file
```

## Key Components

### Data Processing
- Daily price data retrieval
- Log returns calculation
- Price normalization
- Sequence creation for LSTM

### Model Architecture
- EGARCH(1,1) for volatility modeling
- Bidirectional LSTM with attention
- Multi-head attention mechanism
- Multi-task learning setup

### Trading Logic
- Mean reversion based on price spreads
- Volatility-adjusted position sizing
- Transaction cost consideration
- Dynamic threshold adjustment

### Performance Metrics
- Total return
- Sharpe ratio
- Maximum drawdown
- Transaction cost analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Alpaca API for market data
- TensorFlow team for the deep learning framework
- ARCH package developers for GARCH implementation

[List of main packages, e.g., pandas, numpy, matplotlib]

For a complete list, refer to requirements.txt.
