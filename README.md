# Indian Stock Price Predictor

Predict **NSE / BSE** stock prices using technical indicators and machine learning. Backtest the strategy and get next-period forecasts.

## Features

- **Real-time data** from Yahoo Finance for NSE/BSE stocks
- **12+ Technical Indicators**: SMA, EMA, MACD, Bollinger Bands, RSI, Stochastic Oscillator, ATR, OBV, VWAP
- **ML Prediction**: Gradient Boosting / Random Forest with walk-forward validation
- **Backtesting**: Compares strategy returns vs buy-and-hold with Sharpe ratio, directional accuracy, MAE, RMSE, R², MAPE
- **Interactive Charts**: Candlestick + Bollinger Bands, MACD, RSI panels; actual vs predicted overlays
- **Market-Aware**: Predicts next 15-min price when market is open, next-day close when closed
- **Signal Dashboard**: Real-time indicator signal summary (Buy / Sell / Neutral)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| Data | yfinance |
| Indicators | ta (Technical Analysis) |
| ML | scikit-learn |
| Charts | Plotly |

## Disclaimer

This tool is for educational purposes only. Stock market investments are subject to market risk.
