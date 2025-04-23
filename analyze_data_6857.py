import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Profit/Lossと「Time, Date, Session」三つの要素それぞれに対して相関を分析する

# データを読み込む（ファイル名を適宜変更してください）
# 例: data.csv というCSVファイルを読み込む
data = pd.read_csv('trade_data_6857_60days.csv')

# データの先頭を確認
print(data.head())

# 必要な列が存在するか確認
required_columns = ['Time', 'Date', 'Profit/Loss', 'Session']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"データに必要な列が存在しません: {required_columns}")

# TimeとProfit/Lossの関係性を分析
# Timeを数値に変換（例: HH:MM形式を分単位に変換）
data['time_minutes'] = data['Time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

# Dateを数値に変換（例: 日付を日数に変換）
data['date_numeric'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)

# TimeとProfit/Lossの相関を計算
time_corr, time_pval = pearsonr(data['time_minutes'], data['Profit/Loss'])
print(f"TimeとProfit/Lossのピアソン相関: {time_corr}, p値: {time_pval}")

# DateとProfit/Lossの相関を計算
date_corr, date_pval = pearsonr(data['date_numeric'], data['Profit/Loss'])
print(f"DateとProfit/Lossのピアソン相関: {date_corr}, p値: {date_pval}")

# SessionごとのProfit/Lossの平均を計算
session_means = data.groupby('Session')['Profit/Loss'].mean()
print("\nSessionごとのProfit/Lossの平均:")
print(session_means)

# 可視化
plt.figure(figsize=(18, 6))

# TimeとProfit/Lossの散布図
plt.subplot(1, 3, 1)
sns.scatterplot(x='time_minutes', y='Profit/Loss', data=data)
plt.title('Time vs Profit/Loss')
plt.xlabel('Time (minutes)')
plt.ylabel('Profit/Loss')

# DateとProfit/Lossの散布図
plt.subplot(1, 3, 2)
sns.scatterplot(x='date_numeric', y='Profit/Loss', data=data)
plt.title('Date vs Profit/Loss')
plt.xlabel('Date (numeric)')
plt.ylabel('Profit/Loss')

# SessionごとのProfit/Lossの棒グラフ
plt.subplot(1, 3, 3)
sns.barplot(x=session_means.index, y=session_means.values)
plt.title('Session vs Average Profit/Loss')
plt.xlabel('Session')
plt.ylabel('Average Profit/Loss')

plt.tight_layout()
plt.show()

# Profit/LossのヒストグラムとKDEプロット
plt.figure(figsize=(8, 6))
sns.histplot(data['Profit/Loss'], kde=True, bins=20)
plt.title('Distribution of Profit/Loss')
plt.xlabel('Profit/Loss')
plt.ylabel('Frequency')
plt.show()

# 結果の解釈
if time_pval < 0.05:
    print(f"TimeとProfit/Lossの間には統計的に有意な相関があります (相関係数: {time_corr})")
else:
    print("TimeとProfit/Lossの間に統計的に有意な相関はありません。")

if date_pval < 0.05:
    print(f"DateとProfit/Lossの間には統計的に有意な相関があります (相関係数: {date_corr})")
else:
    print("DateとProfit/Lossの間に統計的に有意な相関はありません。")