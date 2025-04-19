import pandas as pd
from scipy.stats import ttest_ind

# CSVファイルを読み込む
comparison_data = pd.read_csv("comparison_data.csv")
ipo_data = pd.read_csv("ipo_detailed_analysis.csv")

# 比較する列を指定
columns_to_compare = ["first_open_price", "price_10_days", "price_30_days", "price_60_days", "price_90_days", "price_180_days"]

# 値動きを計算するための列ペアを作成
column_pairs = [(columns_to_compare[i], columns_to_compare[i + 1]) for i in range(len(columns_to_compare) - 1)]

# 結果を格納するリスト
comparison_results = []

# 各列ペアについて変化率を計算し、統計的な比較を実施
for (col_start, col_end) in column_pairs:
    # 基準群の変化率を計算
    comparison_values = (comparison_data[col_end] - comparison_data[col_start]) / comparison_data[col_start]
    comparison_values = comparison_values.dropna()

    # 比較対象群の変化率を計算
    ipo_values = (ipo_data[col_end] - ipo_data[col_start]) / ipo_data[col_start]
    ipo_values = ipo_values.dropna()

    # 平均値、標準偏差を計算
    comparison_mean = comparison_values.mean()
    comparison_std = comparison_values.std()

    ipo_mean = ipo_values.mean()
    ipo_std = ipo_values.std()

    # t検定を実施
    t_stat, p_value = ttest_ind(comparison_values, ipo_values, equal_var=False)

    # 結果をリストに追加
    comparison_results.append({
        "Metric": f"{col_start} -> {col_end}",
        "Comparison_Mean": comparison_mean,
        "Comparison_Std": comparison_std,
        "IPO_Mean": ipo_mean,
        "IPO_Std": ipo_std,
        "T-Statistic": t_stat,
        "P-Value": p_value
    })

# 結果をデータフレームに変換
results_df = pd.DataFrame(comparison_results)

# 結果を表示
print(results_df)

# 結果をCSVに保存
results_df.to_csv("statistical_comparison_results.csv", index=False)