import pandas as pd
import numpy as np

# 1. データの読み込み
file_path = "ipo_open_close_180_unadjusted.csv"  # ファイルパスを指定
data = pd.read_csv(file_path)

# 2. 必要な列を抽出 (col5からcol364)
opening_prices = data.iloc[:, 4:363:2]  # 始値 (col5, col7, col9, ..., col363)
closing_prices = data.iloc[:, 5:364:2]  # 終値 (col6, col8, col10, ..., col364)

# 3. 欠損値の確認と処理
if opening_prices.isnull().values.any() or closing_prices.isnull().values.any():
    print("欠損値が存在します。欠損値を補完します。")
    opening_prices = opening_prices.fillna(method='ffill').fillna(method='bfill')
    closing_prices = closing_prices.fillna(method='ffill').fillna(method='bfill')

# 4. トレード期間の設定
trade_periods = [10, 60, 120, 180]  # トレード期間 (日数)
returns_by_period = {period: [] for period in trade_periods}  # 各期間のリターンを格納

# 5. 各銘柄ごとにリターンを計算
for i in range(opening_prices.shape[0]):  # 各銘柄ごとに処理
    open_prices = opening_prices.iloc[i, :].values
    close_prices = closing_prices.iloc[i, :].values

    for period in trade_periods:
        if len(open_prices) >= period:  # データが十分にある場合のみ計算
            # 手数料を考慮したリターンを計算
            raw_returns = (open_prices[:period] - close_prices[:period])  # 始値 - 終値
            fees = open_prices[:period] * 0.01  # 手数料は始値の1%
            net_returns = raw_returns - fees  # 手数料を差し引いたリターン
            total_return = np.sum(net_returns)  # 総リターンを計算
            returns_by_period[period].append(total_return)

# 6. 結果を集計
for period, returns in returns_by_period.items():
    returns_series = pd.Series(returns)
    print(f"トレード期間: {period}日")
    print(f"総リターンの基本統計量:\n{returns_series.describe()}")
    print("-" * 50)

# 7. 結果を可視化 (オプション)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for period, returns in returns_by_period.items():
    plt.hist(returns, bins=30, alpha=0.5, label=f"{period} Days")
plt.title("Distribution of Returns by Trade Period (Net of Fees)")
plt.xlabel("Total Return")
plt.ylabel("Frequency")
plt.legend()
plt.show()