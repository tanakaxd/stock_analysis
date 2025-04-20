import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, levene, shapiro
import matplotlib.pyplot as plt
import seaborn as sns

# CSVファイルを読み込む
comparison_data = pd.read_csv("comparison_data.csv")
ipo_data = pd.read_csv("ipo_detailed_analysis.csv")

# 比較する列を指定（col5-184: 日ごとの終値）
comparison_prices = comparison_data.iloc[:, 4:184]
ipo_prices = ipo_data.iloc[:, 4:184]

# 日ごとの変化率を計算
comparison_returns = comparison_prices.pct_change(axis=1).iloc[:, 1:]  # 基準群の変化率
ipo_returns = ipo_prices.pct_change(axis=1).iloc[:, 1:]  # 比較対象群の変化率

# 欠損値を削除
comparison_returns = comparison_returns.dropna()
ipo_returns = ipo_returns.dropna()

# 基本統計量の計算
comparison_mean = comparison_returns.mean().mean()
comparison_std = comparison_returns.std().mean()
ipo_mean = ipo_returns.mean().mean()
ipo_std = ipo_returns.std().mean()

print(f"Comparison Group - Mean: {comparison_mean}, Std: {comparison_std}")
print(f"IPO Group - Mean: {ipo_mean}, Std: {ipo_std}")

# t検定（平均の差の検定）
t_stat, p_value = ttest_ind(comparison_returns.values.flatten(), ipo_returns.values.flatten(), equal_var=False)
print(f"T-Statistic: {t_stat}, P-Value: {p_value}")

# 分散の差の検定（Levene検定）
levene_stat, levene_p = levene(comparison_returns.values.flatten(), ipo_returns.values.flatten())
print(f"Levene Test - Statistic: {levene_stat}, P-Value: {levene_p}")

# 正規性の検定（Shapiro-Wilk検定）
shapiro_comparison_stat, shapiro_comparison_p = shapiro(comparison_returns.values.flatten())
shapiro_ipo_stat, shapiro_ipo_p = shapiro(ipo_returns.values.flatten())
print(f"Shapiro-Wilk Test (Comparison) - Statistic: {shapiro_comparison_stat}, P-Value: {shapiro_comparison_p}")
print(f"Shapiro-Wilk Test (IPO) - Statistic: {shapiro_ipo_stat}, P-Value: {shapiro_ipo_p}")

# ヒストグラムとカーネル密度推定（KDE）で分布を可視化
plt.figure(figsize=(12, 6))
sns.kdeplot(comparison_returns.values.flatten(), label="Comparison Group", shade=True)
sns.kdeplot(ipo_returns.values.flatten(), label="IPO Group", shade=True)
plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Density")
plt.legend()
plt.show()

# ボックスプロットで分布を比較
plt.figure(figsize=(12, 6))
sns.boxplot(data=[comparison_returns.values.flatten(), ipo_returns.values.flatten()], notch=True)
plt.xticks([0, 1], ["Comparison Group", "IPO Group"])
plt.title("Boxplot of Daily Returns")
plt.ylabel("Daily Return")
plt.show()