import yfinance as yf
import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 株価データの取得（例: Appleの株価）
ticker = "AAPL"
data = yf.download(ticker, start="2024-01-01", end="2025-04-22", auto_adjust=False)

# データが空でないか確認
if data.empty:
    raise ValueError(f"No data retrieved for ticker {ticker}")

# 欠損値を処理（必要に応じて）
data = data.ffill()  # 前方補完

# numpy配列に変換し、1Dかつfloat64に
close = data["Close"].to_numpy().flatten().astype(np.float64)
high = data["High"].to_numpy().flatten().astype(np.float64)
low = data["Low"].to_numpy().flatten().astype(np.float64)

# デバッグ：配列の形状と型を確認
print(f"Close array shape: {close.shape}, dtype: {close.dtype}")
print(f"First few values: {close[:5]}")

# 1. 単純移動平均（SMA）と指数平滑移動平均（EMA）の計算
sma = talib.SMA(close, timeperiod=20)  # 20日SMA
ema = talib.EMA(close, timeperiod=20)  # 20日EMA

# 2. ボリンジャーバンドの計算
upper_band, middle_band, lower_band = talib.BBANDS(
    close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
)  # matype=0はSMA

# 3. RSIの計算
rsi = talib.RSI(close, timeperiod=14)  # 14日RSI

# 4. データフレームに結果を統合
df = pd.DataFrame({
    "Close": data["Close"].squeeze(),  # 1次元に変換
    "SMA": sma,
    "EMA": ema,
    "Upper Band": upper_band,
    "Middle Band": middle_band,
    "Lower Band": lower_band,
    "RSI": rsi
}, index=data.index)

# 5. 可視化
plt.figure(figsize=(14, 10))

# 株価と移動平均、ボリンジャーバンドのプロット
plt.subplot(2, 1, 1)
plt.plot(df["Close"], label="Close Price", color="black")
plt.plot(df["SMA"], label="20-day SMA", color="blue")
plt.plot(df["EMA"], label="20-day EMA", color="green")
plt.plot(df["Upper Band"], label="Upper Bollinger Band", color="red", linestyle="--")
plt.plot(df["Middle Band"], label="Middle Bollinger Band", color="orange", linestyle="--")
plt.plot(df["Lower Band"], label="Lower Bollinger Band", color="red", linestyle="--")
plt.title(f"{ticker} Stock Price with SMA, EMA, and Bollinger Bands")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

# RSIのプロット
plt.subplot(2, 1, 2)
plt.plot(df["RSI"], label="RSI (14)", color="purple")
plt.axhline(70, color="red", linestyle="--", label="Overbought (70)")
plt.axhline(30, color="green", linestyle="--", label="Oversold (30)")
plt.title("Relative Strength Index (RSI)")
plt.xlabel("Date")
plt.ylabel("RSI")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("technical_analysis.png")
plt.close()