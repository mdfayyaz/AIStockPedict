"""
Indian Stock Price Prediction App v2.0
Enhanced version with stock search, news, and professional UI
"""
import datetime
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import pytz
import streamlit as st
import ta
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

IST = pytz.timezone("Asia/Kolkata")

# Extended list of Indian stocks (NSE) with more popular options
INDIAN_STOCKS = {
    "Reliance Industries": "RELIANCE",
    "TCS": "TCS",
    "HDFC Bank": "HDFCBANK",
    "Infosys": "INFY",
    "ICICI Bank": "ICICIBANK",
    "SBI": "SBIN",
    "Bharti Airtel": "BHARTIARTL",
    "ITC": "ITC",
    "Kotak Mahindra Bank": "KOTAKBANK",
    "HUL": "HINDUNILVR",
    "Wipro": "WIPRO",
    "Asian Paints": "ASIANPAINT",
    "Maruti Suzuki": "MARUTI",
    "Bajaj Finance": "BAJFINANCE",
    "Tata Motors": "TATAMOTORS",
    "Tata Steel": "TATASTEEL",
    "Adani Enterprises": "ADANIENT",
    "Axis Bank": "AXISBANK",
    "Bajaj Auto": "BAJAJFINSV", 
    "Sun Pharma": "SUNPHARMA",
    "NTPC": "NTPC",
    "Power Grid": "POWERGRID",
    "IndusInd Bank": "INDUSINDBK",
    "Larsen & Toubro": "LT",
    "Mahindra & Mahindra": "M&M",
    "Apollo Hospitals": "APOLLOHOSP",
    "Titan Company": "TITAN",
    "UltraTech Cement": "ULTRACEMCO",
    "Nestle India": "NESTLEIND",
    "Hero MotoCorp": "HEROMOTOCO",
    "Lupin": "LUPIN",
    "Cipla": "CIPLA",
    "Britannia": "BRITANNIA",
    "Divi's Laboratories": "DIVISLAB",
    "Dr. Reddy's": "DRREDDY",
    "HCL Technologies": "HCLTECH",
    "Tech Mahindra": "TECHM",
    "Zomato": "ZOMATO",
    "Jio Financial Services": "JIOFIN",
    "Tata Power": "TATAPOWER",
    "Suzlon Energy": "SUZLON",
    "IRFC": "IRFC",
    "IREDA": "IREDA",
    "RVNL": "RVNL",
    "Adani Ports": "ADANIPORTS",
    "Adani Power": "ADANIPOWER",
    "Adani Green": "ADANIGREEN",
    "Avenue Supermarts (DMart)": "DMART",
    "Yes Bank": "YESBANK",
    "Punjab National Bank": "PNB",
    "Hindustan Aeronautics (HAL)": "HAL",
    "Bharat Electronics (BEL)": "BEL",
    "Varun Beverages": "VBL",
    "One97 Communications (Paytm)": "PAYTM",
    "MRF": "MRF",
    "Nykaa (FSN E-Commerce)": "NYKAA",
    "NIFTY 50 Index": "^NSEI",
    "SENSEX Index": "^BSESN",
}


def is_market_open() -> bool:
    now_ist = datetime.datetime.now(IST)
    weekday = now_ist.weekday()
    if weekday >= 5: 
        return False
    market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now_ist <= market_close


@st.cache_data(ttl=60) # Reduced cache time for more current data
def fetch_daily_data(ticker: str, period_days: int = 30) -> pd.DataFrame:
    extra_warmup = 60
    total_days = period_days + extra_warmup
    # Fetch data up to today, period parameter is more reliable for recent data
    df = yf.download(ticker, period=f"{total_days}d", interval="1d", progress=False)
    if df.empty: 
        return df
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)
    df.index = pd.DatetimeIndex(df.index)
    return df


@st.cache_data(ttl=60)
def fetch_intraday_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="60d", interval="15m", progress=False)
    if df.empty: 
        return df
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)
    df.index = pd.DatetimeIndex(df.index)
    return df


@st.cache_data(ttl=3600)
def fetch_stock_news(ticker: str) -> list:
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news[:6] if news else []
    except Exception:
        return []

def get_realtime_price(ticker_symbol: str):
    """Fetch the absolute latest price using yfinance info dictionary."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Try fast_info first (usually faster and more reliable for real-time)
        if hasattr(ticker, 'fast_info'):
            fast_info = ticker.fast_info
            if fast_info and 'lastPrice' in fast_info and not pd.isna(fast_info['lastPrice']):
                return float(fast_info['lastPrice'])
                
        # Fallback to info dictionary
        info = ticker.info
        if info and 'currentPrice' in info and not pd.isna(info['currentPrice']):
            return float(info['currentPrice'])
            
    except Exception:
        pass
        
    return None


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"].astype(float)
    
    df["SMA_5"] = ta.trend.sma_indicator(close, window=5)
    df["SMA_10"] = ta.trend.sma_indicator(close, window=10)
    df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    df["EMA_5"] = ta.trend.ema_indicator(close, window=5)
    df["EMA_10"] = ta.trend.ema_indicator(close, window=10)
    df["EMA_20"] = ta.trend.ema_indicator(close, window=20)
    
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()
    df["BB_Pct"] = bb.bollinger_pband()
    
    df["RSI_14"] = ta.momentum.rsi(close, window=14)
    
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    
    df["ATR_14"] = ta.volatility.average_true_range(high, low, close, window=14)
    df["OBV"] = ta.volume.on_balance_volume(close, volume)
    
    typical_price = (high + low + close) / 3
    df["VWAP"] = (typical_price * volume).cumsum() / volume.cumsum()
    
    df["Returns"] = close.pct_change()
    df["Log_Returns"] = np.log(close / close.shift(1))
    df["High_Low_Range"] = high - low
    df["Close_Open_Diff"] = close - df["Open"]
    
    return df


def prepare_features(df: pd.DataFrame, target_col: str = "Close"):
    df = add_technical_indicators(df)
    df["Target"] = df[target_col].shift(-1)
    df.dropna(inplace=True)
    
    feature_cols = [
        "SMA_5", "SMA_10", "SMA_20", "EMA_5", "EMA_10", "EMA_20",
        "MACD", "MACD_Signal", "MACD_Hist", "BB_Upper", "BB_Middle", "BB_Lower", "BB_Width", "BB_Pct",
        "RSI_14", "Stoch_K", "Stoch_D", "ATR_14", "OBV", "VWAP", "Returns", "Log_Returns",
        "High_Low_Range", "Close_Open_Diff", "Close", "Open", "High", "Low", "Volume",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    x = df[feature_cols].values
    y = df["Target"].values
    dates = df.index
    return x, y, dates, feature_cols, df


def train_and_predict(x, y, model_type="GradientBoosting"):
    scaler = StandardScaler()
    if model_type != "RandomForest":
        model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42) 
    else: 
        model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    
    n_splits = min(5, len(x) // 3)
    if n_splits < 2: 
        n_splits = 2
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    all_preds = np.full(len(y), np.nan)
    all_actuals = np.full(len(y), np.nan)
    
    for train_idx, test_idx in tscv.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        x_train_sc = scaler.fit_transform(x_train)
        x_test_sc = scaler.transform(x_test)
        model.fit(x_train_sc, y_train)
        all_preds[test_idx] = model.predict(x_test_sc)
        all_actuals[test_idx] = y_test
    
    x_all_sc = scaler.fit_transform(x)
    model.fit(x_all_sc, y)
    next_pred = model.predict(x_all_sc[-1:])
    importances = model.feature_importances_
    
    return all_preds, all_actuals, next_pred[0], importances, model, scaler


def backtest_metrics(actuals, preds):
    mask = ~np.isnan(preds) & ~np.isnan(actuals)
    a, p = actuals[mask], preds[mask]
    if len(a) == 0: 
        return {}
    
    mae = mean_absolute_error(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    r2 = r2_score(a, p)
    mape = np.mean(np.abs((a - p) / a)) * 100
    
    actual_dir = np.diff(a) > 0
    pred_dir = np.diff(p) > 0
    dir_acc = np.mean(actual_dir == pred_dir) * 100 if len(actual_dir) > 0 else 0
    
    strategy_returns = []
    buy_hold_returns = []
    for i in range(1, len(a)):
        ret = (a[i] - a[i - 1]) / a[i - 1]
        buy_hold_returns.append(ret)
        strategy_returns.append(ret if p[i] > a[i - 1] else -ret)
    
    strategy_returns = np.array(strategy_returns) if strategy_returns else np.array([0])
    buy_hold_returns = np.array(buy_hold_returns) if buy_hold_returns else np.array([0])
    
    cum_strategy = (1 + strategy_returns).cumprod()[-1] - 1
    cum_buyhold = (1 + buy_hold_returns).cumprod()[-1] - 1
    if np.std(strategy_returns) > 0:
        sharpe = (np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252))
    else: 
        sharpe = 0
    
    return {
        "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R²": round(r2, 4),
        "MAPE %": round(mape, 2), "Dir Acc %": round(dir_acc, 2),
        "Strategy %": round(float(cum_strategy * 100), 2),
        "B&H %": round(float(cum_buyhold * 100), 2), "Sharpe": round(float(sharpe), 2),
    }


def get_indicator_summary(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    signals = {}
    
    rsi = latest.get("RSI_14", 50)
    if pd.isna(rsi): 
        signals["RSI"] = "🟡 Neutral"
    elif rsi > 70: 
        signals["RSI"] = "🔴 Overbought"
    elif rsi < 30: 
        signals["RSI"] = "🟢 Oversold"
    else: 
        signals["RSI"] = "🟡 Neutral"
    
    macd = latest.get("MACD", 0)
    macd_signal = latest.get("MACD_Signal", 0)
    if pd.isna(macd) or pd.isna(macd_signal): 
        signals["MACD"] = "🟡 Neutral"
    elif macd > macd_signal: 
        signals["MACD"] = "🟢 Bullish"
    else: 
        signals["MACD"] = "🔴 Bearish"
    
    close = latest.get("Close", 0)
    sma20 = latest.get("SMA_20", 0)
    if pd.isna(sma20): 
        signals["SMA 20"] = "🟡 Neutral"
    elif close > sma20: 
        signals["SMA 20"] = "🟢 Bullish"
    else: 
        signals["SMA 20"] = "🔴 Bearish"
    
    bb_upper = latest.get("BB_Upper", 0)
    bb_lower = latest.get("BB_Lower", 0)
    if pd.isna(bb_upper) or pd.isna(bb_lower): 
        signals["Bollinger Bands"] = "🟡 Neutral"
    elif close > bb_upper: 
        signals["Bollinger Bands"] = "🔴 Overbought"
    elif close < bb_lower: 
        signals["Bollinger Bands"] = "🟢 Oversold"
    else: 
        signals["Bollinger Bands"] = "🟡 Neutral"
    
    return signals


def plot_backtest(dates, actuals, preds, title="Backtest"):
    mask = ~np.isnan(preds)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[mask], y=actuals[mask], mode="lines+markers", name="Actual", line=dict(color="#2196F3", width=2)))
    fig.add_trace(go.Scatter(x=dates[mask], y=preds[mask], mode="lines+markers", name="Predicted", line=dict(color="#FF5722", width=2, dash="dash")))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price (₹)", template="plotly_dark", height=500, hovermode="x unified")
    return fig


def plot_indicators(df: pd.DataFrame):
    df_plot = add_technical_indicators(df).dropna()
    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.22, 0.23])
    
    fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot["Open"], high=df_plot["High"], low=df_plot["Low"], close=df_plot["Close"], name="OHLC"), row=1, col=1)
    
    for col_name, color in [("BB_Upper", "rgba(255,193,7,0.5)"), ("BB_Middle", "rgba(255,193,7,0.8)"), ("BB_Lower", "rgba(255,193,7,0.5)")]:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col_name], mode="lines", name=col_name, line=dict(color=color, width=1)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MACD"], name="MACD", line=dict(color="#26C6DA")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MACD_Signal"], name="Signal", line=dict(color="#FF7043")), row=2, col=1)
    
    colors = ["green" if v >= 0 else "red" for v in df_plot["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["MACD_Hist"], name="Histogram", marker=dict(color=colors)), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["RSI_14"], name="RSI", line=dict(color="#AB47BC", width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(template="plotly_dark", height=800, showlegend=True, hovermode="x unified", dragmode="zoom", clickmode="event+select")
    return fig


def main():
    st.set_page_config(page_title="Indian Stock Predictor", page_icon="📈", layout="wide")
    
    # Hide default top padding for a sleeker UI
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("🇮🇳 Indian Stock Price Predictor")
    
    with st.sidebar:
        st.header("🔍 Search")
        
        stock_options = [""] + sorted(list(INDIAN_STOCKS.keys()))
        
        selected_stock_name = st.selectbox(
            "Select Stock", 
            options=stock_options, 
            index=0,
            label_visibility="collapsed", # Hide default label for cleaner look
            placeholder="Type to search (e.g., Reliance, TCS)",
            key="stock_selector"
        )
        
        analyze_button = st.button("Analyze", type="primary", use_container_width=True)
        
        st.divider()
        st.header("⚙️ Settings")
        period_days = st.slider("Look-back period (days)", 15, 90, 30)
        model_type = st.radio("ML Model", ["GradientBoosting", "RandomForest"], index=0)
        
        market_status = is_market_open()
        if market_status:
            st.success("🟢 Market OPEN")
        else:
            st.info("🔴 Market CLOSED")

    # --- Professional Default Welcome Screen ---
    if not selected_stock_name or not analyze_button:
        st.markdown(
            """
            <style>
            .welcome-title {
                font-size: 1.8em;
                font-weight: 600;
                color: #4CAF50;
                margin-bottom: 5px;
            }
            .welcome-subtitle {
                font-size: 1em;
                color: #888888;
                margin-bottom: 30px;
            }
            .news-header {
                font-size: 1.2em;
                font-weight: 600;
                color: #E0E0E0;
                margin-top: 10px;
                margin-bottom: 15px;
                border-bottom: 1px solid #444;
                padding-bottom: 5px;
            }
            .news-link {
                text-decoration: none;
                color: #2196F3;
                font-weight: 500;
            }
            .news-link:hover {
                text-decoration: underline;
            }
            .developer-credit {
                margin-top: 40px;
                font-size: 0.85em;
                color: #666666;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="welcome-title">AI-Powered Market Insights</div>', unsafe_allow_html=True)
        st.markdown('<div class="welcome-subtitle">Search and select a stock from the sidebar to generate technical analysis and ML predictions.</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="news-header">📰 Latest Market Updates (NIFTY 50)</div>', unsafe_allow_html=True)
        
        with st.spinner("Loading market updates..."):
            market_news = fetch_stock_news("^NSEI")
            
        if not market_news:
            st.info("Market updates are currently unavailable.")
        else:
            cols = st.columns(2)
            for idx, article in enumerate(market_news[:6]):
                with cols[idx % 2]:
                    with st.container(border=True):
                        title = article.get('title', 'No Title')
                        publisher = article.get('publisher', 'Unknown')
                        link = article.get('link', '#')
                        st.markdown(f"<a href='{link}' target='_blank' class='news-link'>{title}</a>", unsafe_allow_html=True)
                        st.caption(f"{publisher}")

        st.markdown(
            """
            <div class="developer-credit">
                Developed by <strong>Md. Fayazuddin</strong> for training and educational purposes.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # User clicked Analyze with a selected stock
    search_query_clean = selected_stock_name.strip().upper()
    
    # Resolve ticker symbol
    ticker = search_query_clean
    search_query_display = selected_stock_name # Use the full name for display
    
    # Find the actual ticker symbol from our INDIAN_STOCKS dictionary
    if selected_stock_name in INDIAN_STOCKS:
        ticker = INDIAN_STOCKS[selected_stock_name]
    else:
        for name, symbol in INDIAN_STOCKS.items():
            if search_query_clean == symbol.upper():
                ticker = symbol
                search_query_display = name
                break
    
    # Auto-append .NS for Indian stocks if not already an index or explicitly .BO
    if not ticker.endswith(".NS") and not ticker.endswith(".BO") and not ticker.startswith("^"):
        ticker = ticker + ".NS"

    with st.spinner(f"Fetching latest data for {ticker}..."):
        daily_df = fetch_daily_data(ticker, period_days)
    
    if daily_df.empty:
        st.error(f"No data found for symbol: {ticker}. Please check the symbol or try another one.")
        return
    
    # Ensure 'Close' column exists and has valid data
    if 'Close' not in daily_df.columns or daily_df['Close'].isnull().all():
        st.error(f"No valid closing price data found for {ticker}. The data may be unavailable or corrupted.")
        return

    # Data Processing for Display
    last_valid_close_idx = daily_df['Close'].last_valid_index()
    latest_daily = daily_df.loc[last_valid_close_idx]
    
    # Attempt to get real-time price using the dedicated function
    realtime_price = get_realtime_price(ticker)
    
    # The current price is real-time if available, otherwise the last close from history
    current_price = realtime_price if realtime_price is not None else latest_daily['Close']

    # Find previous close for change calculation
    valid_closes = daily_df['Close'].dropna()
    if len(valid_closes) < 2:
        st.warning(f"Only one or no valid historical closing prices found for {ticker}. Cannot calculate daily change.")
        prev_close = latest_daily['Close'] 
    else:
        # If we successfully grabbed a real-time price, the "previous" close is the last full day's close
        # Otherwise, the "current" price is the last full day's close, and "previous" is the day before that.
        if realtime_price is not None:
             # If we have a real-time price, the previous close is the last recorded daily close
             prev_close = valid_closes.iloc[-1] if len(valid_closes) > 0 else current_price
        else:
             # If no real-time, current_price is latest_daily['Close'], so prev_close is the one before that
             prev_close = valid_closes.iloc[-2] if len(valid_closes) > 1 else current_price

    st.subheader(f"📊 {search_query_display} ({ticker}) Latest Prices")
    
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0 
    
    col1, col2, col3, col4 = st.columns(4)
    # Highlight if using real-time
    price_label = "Current Price (Real-time)" if realtime_price is not None else "Close Price"
    col1.metric(price_label, f"₹{current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    col2.metric("High (Last close)", f"₹{latest_daily['High']:.2f}")
    col3.metric("Low (Last close)", f"₹{latest_daily['Low']:.2f}")
    col4.metric("Volume", f"{latest_daily['Volume']:,.0f}")
    
    st.divider()
    
    # ML Prediction
    x, y, dates, feat_cols, feat_df = prepare_features(daily_df, "Close")
    if len(x) < 6:
        st.warning("Not enough historical data to run the ML model. Please try a longer look-back period.")
        return
    
    preds, actuals, next_day_pred, importances, model, scaler = train_and_predict(x, y, model_type)
    
    st.subheader("💡 Next Price Prediction")
    if pd.isna(next_day_pred):
        st.warning("Could not generate a price prediction.")
    else:
        # Predict relative to current known price
        diff_d = next_day_pred - current_price
        pct_d = diff_d / current_price * 100
        direction = "📈 Bullish" if diff_d > 0 else "📉 Bearish"
        st.success(f"**Next Trading Day Expected Close:** ₹{next_day_pred:.2f} | {diff_d:+.2f} ({pct_d:+.2f}%) — {direction}")
    
    st.divider()
    
    st.subheader("📊 Daily Close Prediction & Backtest")
    st.plotly_chart(plot_backtest(dates, actuals, preds, "Backtest — Daily Close"), use_container_width=True, key="backtest")
    
    st.divider()
    
    st.subheader("📋 Backtest Metrics")
    metrics = backtest_metrics(actuals, preds)
    if metrics:
        cols = st.columns(4)
        for i, (k, v) in enumerate(metrics.items()):
            cols[i % 4].metric(k, v)
    
    st.divider()
    
    st.subheader("🚦 Indicator Signal Summary")
    signals = get_indicator_summary(feat_df)
    sig_cols = st.columns(len(signals))
    for i, (k, v) in enumerate(signals.items()):
        with sig_cols[i]:
            st.markdown(f"**{k}**")
            st.markdown(f"*{v}*")
            
    st.divider()
    
    st.subheader("📈 Technical Analysis")
    st.plotly_chart(plot_indicators(daily_df), use_container_width=True, key="indicators")
    
    st.divider()
    
    st.subheader("📰 Latest News & Updates")
    news = fetch_stock_news(ticker)
    if not news:
        st.info("No news available for this stock at the moment.")
    else:
        cols = st.columns(2)
        for idx, article in enumerate(news[:4]):
            with cols[idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{article.get('title', 'No Title')}**")
                    st.caption(article.get('publisher', 'Unknown'))
                    if 'link' in article:
                        st.markdown(f"[Read More →]({article['link']})")
    
    st.divider()
    st.markdown('<div style="font-size:10px;color:#888;background:#f5f5f5;padding:8px;border-left:3px solid #ff9800;margin-top:15px"><strong style="color:#ff9800">⚠️ Disclaimer:</strong> For educational purposes only. Do not use as a sole basis for trading. Consult SEBI-registered professionals before making financial decisions. Past performance does not guarantee future results.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
