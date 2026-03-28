import yfinance as yf
ticker = "RELIANCE.NS"
df = yf.download(ticker, period="1mo", progress=False)
print(df.tail())
