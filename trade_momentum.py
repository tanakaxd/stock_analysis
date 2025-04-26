import yfinance as yf
import pandas as pd
import numpy as np

# 株価データの取得（例：Appleの株価）
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
data['Returns'] = data['Close'].pct_change()

# モメンタム戦略のパラメータ
lookback = 20  # 過去20日のリターンを確認
holding_period = 5  # 5日間保有

# モメンタムの計算（過去20日の累積リターン）
data['Momentum'] = data['Close'].pct_change(lookback)

# シグナルの生成：モメンタムが正なら買い、負なら売り
data['Signal'] = np.where(data['Momentum'] > 0, 1, -1)

# リターンの計算（5日後のリターン）
data['Future_Return'] = data['Close'].shift(-holding_period).pct_change(holding_period)

# 戦略リターンの計算
data['Strategy_Return'] = data['Signal'] * data['Future_Return']

# 取引コスト（仮に0.1%）
transaction_cost = 0.001
data['Strategy_Return'] = data['Strategy_Return'] - transaction_cost * data['Signal'].diff().abs()

# 結果の評価
cumulative_return = (1 + data['Strategy_Return']).cumprod()
sharpe_ratio = data['Strategy_Return'].mean() / data['Strategy_Return'].std() * np.sqrt(252)

print(f"累積リターン: {cumulative_return[-1]:.2f}")
print(f"シャープレシオ: {sharpe_ratio:.2f}")

# プロット
import matplotlib.pyplot as plt
plt.plot(cumulative_return)
plt.title(f"{ticker} Momentum Strategy")
plt.show()