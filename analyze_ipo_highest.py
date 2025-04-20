import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# 1. データの読み込み
file_path = "ipo_detailed_analysis.csv"  # ファイルパスを指定
data = pd.read_csv(file_path)

# 2. 必要な列を抽出 (col5からcol184)
closing_prices = data.iloc[:, 4:184]  # col5-184に対応

# 3. 欠損値の確認と処理
if closing_prices.isnull().values.any():
    print("欠損値が存在します。欠損値を補完します。")
    closing_prices = closing_prices.fillna(method='ffill').fillna(method='bfill')

# 4. 初日の価格を基準に割合に変換
normalized_prices = closing_prices.div(closing_prices.iloc[:, 0], axis=0)  # 初日の価格で割る

# 5. 最高値と局所的な極値を特定
max_days = []  # 各銘柄の最高値の日数を格納
local_max_days = []  # 各銘柄の局所的な極値の日数を格納

for i in range(normalized_prices.shape[0]):  # 各銘柄ごとに処理
    prices = normalized_prices.iloc[i, :].values

    # 最高値の日数を取得
    max_day = np.argmax(prices) + 1  # 日数は1から始まる
    max_days.append(max_day)

    # 局所的な極値（ローカルマックス）を取得
    local_max_indices = argrelextrema(prices, np.greater)[0]  # 局所的な極値のインデックス
    local_max_days.extend(local_max_indices + 1)  # 日数は1から始まる

# 6. 最高値と局所的な極値の日数の分布を集計
max_days_series = pd.Series(max_days)
local_max_days_series = pd.Series(local_max_days)

# 7. 分布を可視化
plt.figure(figsize=(12, 6))
plt.hist(max_days_series, bins=30, alpha=0.7, label="Highest Price Days", color="blue")
plt.hist(local_max_days_series, bins=30, alpha=0.5, label="Local Max Days", color="orange")
plt.title("Distribution of High Price Days (Normalized)")
plt.xlabel("Days Since IPO")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# 8. 傾向の分析
# 最高値の日数の基本統計量
max_days_stats = max_days_series.describe()
print("最高値の日数の基本統計量:\n", max_days_stats)

# 局所的な極値の日数の基本統計量
local_max_days_stats = local_max_days_series.describe()
print("局所的な極値の日数の基本統計量:\n", local_max_days_stats)

# 9. 平均的な価格推移をプロット
mean_prices = normalized_prices.mean(axis=0)  # 各日付の平均価格
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(mean_prices) + 1), mean_prices, label="Average Normalized Price", color="green")
plt.title("Average Normalized Price Trend Over 180 Days")
plt.xlabel("Days Since IPO")
plt.ylabel("Normalized Price")
plt.legend()
plt.show()