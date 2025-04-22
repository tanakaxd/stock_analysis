import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. データの読み込み
file_path = "ipo_open_close_180_unadjusted.csv"  # ファイルパスを指定
data = pd.read_csv(file_path)

# 2. 必要な列を抽出 (col5からcol64)
opening_prices = data.iloc[:, 4:64:2]  # 始値 (col5, col7, col9, ..., col63)
closing_prices = data.iloc[:, 5:65:2]  # 終値 (col6, col8, col10, ..., col64)

# 3. 欠損値の確認と処理
if opening_prices.isnull().values.any() or closing_prices.isnull().values.any():
    print("欠損値が存在します。欠損値を補完します。")
    opening_prices = opening_prices.fillna(method='ffill').fillna(method='bfill')
    closing_prices = closing_prices.fillna(method='ffill').fillna(method='bfill')

# 4. トレード期間の設定
trade_period = 30  # トレード期間を30日に設定
net_returns_with_fees = []  # 手数料ありのネットリターンを格納
net_returns_without_fees = []  # 手数料なしのネットリターンを格納

# 5. 各銘柄ごとにリターンを計算
for i in range(opening_prices.shape[0]):  # 各銘柄ごとに処理
    open_prices = opening_prices.iloc[i, :].values
    close_prices = closing_prices.iloc[i, :].values

    if len(open_prices) >= trade_period:  # データが十分にある場合のみ計算
        # 手数料なしのリターンを計算
        raw_returns = (open_prices[:trade_period] - close_prices[:trade_period])  # 始値 - 終値
        total_return_without_fees = np.sum(raw_returns)  # 手数料なしの総リターンを計算
        net_returns_without_fees.append(total_return_without_fees)

        # 手数料ありのリターンを計算
        fees = open_prices[:trade_period] * 0.01  # 手数料は始値の1%
        net_return_with_fees = np.sum(raw_returns - fees)  # 手数料を差し引いた総リターンを計算
        net_returns_with_fees.append(net_return_with_fees)
    else:
        net_returns_without_fees.append(None)  # データが不足している場合はNoneを格納
        net_returns_with_fees.append(None)  # データが不足している場合はNoneを格納

# 6. 結果をデータフレームに格納
results_df = pd.DataFrame({
    "ticker": data["ticker"],  # 銘柄コード
    "company_name": data["company_name"],  # 会社名
    "net_return_30_days_without_fees": net_returns_without_fees,  # 手数料なしの30日間のネットリターン
    "net_return_30_days_with_fees": net_returns_with_fees  # 手数料ありの30日間のネットリターン
})

# 7. 統計量を計算
print("手数料なしのリターンの統計量:")
print(results_df["net_return_30_days_without_fees"].describe())
print("\n手数料ありのリターンの統計量:")
print(results_df["net_return_30_days_with_fees"].describe())

# 8. 結果をCSVに保存
output_file = "ipo_net_returns_30days_with_and_without_fees.csv"
results_df.to_csv(output_file, index=False)
print(f"計算結果をCSVファイルに保存しました: {output_file}")

# 9. 可視化
plt.figure(figsize=(12, 6))

# 手数料なしのリターン分布
plt.hist(
    results_df["net_return_30_days_without_fees"].dropna(),
    bins=30,
    alpha=0.5,
    label="Without Fees",
    color="blue"
)

# 手数料ありのリターン分布
plt.hist(
    results_df["net_return_30_days_with_fees"].dropna(),
    bins=30,
    alpha=0.5,
    label="With Fees",
    color="orange"
)

# グラフの設定
plt.title("Distribution of 30-Day Net Returns (With and Without Fees)")
plt.xlabel("Net Return")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()