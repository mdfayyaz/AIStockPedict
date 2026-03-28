import yfinance as yf
import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")

def diagnose_reliance_price():
    ticker_symbol = "RELIANCE.NS"
    
    print(f"--- Diagnosing {ticker_symbol} Price ---")

    # Attempt to get real-time info
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        fast_info = ticker.fast_info

        current_time_ist = datetime.datetime.now(IST)
        print(f"Current IST Time: {current_time_ist.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")

        if info and 'currentPrice' in info:
            print(f"yf.Ticker().info['currentPrice']: {info['currentPrice']:.2f}")
        else:
            print("yf.Ticker().info['currentPrice']: Not available")
            
        if fast_info and 'lastPrice' in fast_info:
            print(f"yf.Ticker().fast_info['lastPrice']: {fast_info['lastPrice']:.2f}")
        else:
            print("yf.Ticker().fast_info['lastPrice']: Not available")

        if info and 'previousClose' in info:
            print(f"yf.Ticker().info['previousClose']: {info['previousClose']:.2f}")
        else:
            print("yf.Ticker().info['previousClose']: Not available")

    except Exception as e:
        print(f"Error fetching real-time info: {e}")

    # Fetch daily historical data for the last few days
    try:
        end_date = datetime.datetime.now(IST) + datetime.timedelta(days=1) # Ensure today is included
        start_date = end_date - datetime.timedelta(days=7)
        df_daily = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1d", progress=False)
        print("\n--- Latest Daily Historical Data (1d interval) ---")
        if not df_daily.empty:
            print(df_daily.tail())
        else:
            print("No daily historical data found.")
    except Exception as e:
        print(f"Error fetching daily historical data: {e}")

    # Fetch intraday data for today (if market is open)
    try:
        current_time_ist = datetime.datetime.now(IST)
        market_open_ist = current_time_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close_ist = current_time_ist.replace(hour=15, minute=30, second=0, microsecond=0)

        if market_open_ist <= current_time_ist <= market_close_ist and current_time_ist.weekday() < 5:
            print("\n--- Latest Intraday Historical Data (15m interval, last 1 day) ---")
            df_intraday = yf.download(ticker_symbol, period="1d", interval="15m", progress=False)
            if not df_intraday.empty:
                print(df_intraday.tail())
            else:
                print("No intraday historical data found for today.")
        else:
            print("\nMarket is currently closed or it's a weekend. Intraday data not fetched.")
    except Exception as e:
        print(f"Error fetching intraday data: {e}")

if __name__ == "__main__":
    diagnose_reliance_price()
