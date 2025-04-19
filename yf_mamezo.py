import yfinance as yf

# 銘柄設定
ticker = "202A.T"
stock = yf.Ticker(ticker)

# 配当履歴
dividends = stock.dividends
print("配当履歴:\n", dividends)

# 分割履歴
splits = stock.splits
print("分割履歴:\n", splits)

# 調整済み vs 未調整データの比較（上場日）
data_adjusted = stock.history(start="2024-06-27", end="2024-06-28", auto_adjust=True)
data_unadjusted = stock.history(start="2024-06-27", end="2024-06-28", auto_adjust=False)
print("調整済み初値:", data_adjusted["Open"].iloc[0])
print("未調整初値:", data_unadjusted["Open"].iloc[0])
print("調整済み終値:", data_adjusted["Close"].iloc[0])
print("未調整終値:", data_unadjusted["Close"].iloc[0])