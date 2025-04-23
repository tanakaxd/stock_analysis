import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import seaborn as sns

# 1. 日経先物の2025年5分足データを取得
def fetch_nikkei_futures_data():
    # ticker = "NIY=F"  # 日経先物のティッカー
    # ticker = "^N225"  # 日経平均のティッカー
    ticker = "6857.T"  # 日経平均のティッカー
    df = yf.download(ticker, start="2025-02-23", end="2025-04-22", interval="5m")
    print(f"ticker: {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        print("MultiIndex detected, flattening...")
        df.columns = df.columns.get_level_values(0)
    return df

# 2. エスカリエパターンを検出
def detect_escalier_patterns(data):
    small_candle_threshold = 0.0005  # 0.05%
    large_candle_threshold = 0.001  # 0.1%

    data['change'] = (data['Close'] - data['Open']).abs() / data['Open']
    patterns = []

    for i in range(1, len(data) - 1):
        prev_candle = data.iloc[i - 1]['change']
        current_candle = data.iloc[i]['change']
        next_candle = data.iloc[i + 1]['change']

        # ターンの中心となるローソク足が陽線であることを確認
        is_bullish = data.iloc[i]['Close'] > data.iloc[i]['Open']

        if (
            prev_candle <= small_candle_threshold and
            current_candle >= large_candle_threshold and
            next_candle <= small_candle_threshold and
            is_bullish  # 陽線であることを確認
        ):
            patterns.append(data.iloc[i].name)  # パターンの中心となるローソク足のタイムスタンプ

    return patterns

# 3. トレンド分析
def analyze_trend(data, patterns):
    results = []
    # data = data.copy()  # データをコピーして元のデータを変更しないようにする
    candles_ahead_count = 20  # 何本先のローソク足までを対象として分析するか

    for pattern_time in patterns:
        # パターン後のデータを取得
        pattern_index = data.index.get_loc(pattern_time)
        future_data = data.iloc[pattern_index:pattern_index + candles_ahead_count].copy()

        # 移動平均線
        future_data['SMA'] = future_data['Close'].rolling(window=5).mean()

        # RSI
        delta = future_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        future_data['RSI'] = 100 - (100 / (1 + gain / loss))

        # MACD
        ema12 = future_data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = future_data['Close'].ewm(span=26, adjust=False).mean()
        future_data['MACD'] = ema12 - ema26

        # 線形回帰
        X = np.arange(len(future_data)).reshape(-1, 1)
        y = future_data['Close'].values
        model = LinearRegression().fit(X, y)
        trend_slope = model.coef_[0]

        # トレンド方向を記録
        results.append({
            'pattern_time': pattern_time,
            'trend_slope': trend_slope,
            'RSI': future_data['RSI'].iloc[-1],
            'MACD': future_data['MACD'].iloc[-1]
        })

    return results

def plot_escalier_patterns(data, patterns, trend_analysis):
    for result in trend_analysis:
        pattern_time = result['pattern_time']
        trend_slope = result['trend_slope']
        rsi = result['RSI']
        macd = result['MACD']

        # パターンの中心から一定範囲のデータを取得
        pattern_index = data.index.get_loc(pattern_time)
        plot_data = data.iloc[max(0, pattern_index - 10):pattern_index + 20].copy()

        # 移動平均線を計算
        plot_data['SMA'] = plot_data['Close'].rolling(window=5).mean()

        # RSIとMACDを追加
        plot_data['RSI'] = np.nan
        plot_data['MACD'] = np.nan

        # trend_analysisの結果からRSIとMACDを適用
        if len(plot_data) > 0:
            plot_data.loc[pattern_time:, 'RSI'] = rsi
            plot_data.loc[pattern_time:, 'MACD'] = macd

        # チャートを描画
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # ローソク足チャート（Close価格）
        ax1.plot(plot_data.index, plot_data['Close'], label='Close', color='blue', linewidth=1.5)
        ax1.plot(plot_data.index, plot_data['SMA'], label='SMA (5)', color='orange', linestyle='--')

        # エスカリエパターンの中心をマーカーで表示
        ax1.axvline(x=pattern_time, color='red', linestyle='--', label='Pattern Center')

        ax1.set_title(f"Escalier Pattern at {pattern_time}\nTrend Slope: {trend_slope:.4f}, RSI: {rsi:.2f}, MACD: {macd:.4f}")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend(loc='upper left')

        # RSIを描画
        ax2 = ax1.twinx()
        ax2.plot(plot_data.index, plot_data['RSI'], label='RSI', color='green', linestyle='-.')
        ax2.axhline(y=70, color='gray', linestyle='--', linewidth=0.8)
        ax2.axhline(y=30, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_ylabel("RSI")
        ax2.legend(loc='upper right')

        # MACDを描画
        fig, ax3 = plt.subplots(figsize=(12, 3))
        ax3.plot(plot_data.index, plot_data['MACD'], label='MACD', color='purple')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax3.set_title("MACD")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("MACD")
        ax3.legend()

        # グラフを表示
        plt.tight_layout()
        plt.show()

# メイン処理
if __name__ == "__main__":
    data = fetch_nikkei_futures_data()
    print("データ取得完了")
    # print(data.head())  # デバッグ用出力
    # print(data.tail())  # デバッグ用出力
    # print(data.iloc[70:80])  # デバッグ用出力

    patterns = detect_escalier_patterns(data)
    trend_analysis = analyze_trend(data, patterns)

    # 結果を表示
    for result in trend_analysis:
        print(result)

    # trend_slopeの統計情報を計算
    if trend_analysis:
        trend_slopes = [result['trend_slope'] for result in trend_analysis]

        # 平均値
        average_trend_slope = sum(trend_slopes) / len(trend_slopes)
        # 中央値
        median_trend_slope = np.median(trend_slopes)
        # 標準偏差
        std_trend_slope = np.std(trend_slopes, ddof=1)  # 標本標準偏差

        # 統計的検定（1標本t検定）
        # 片側検定（平均が0より大きいかを検定）
        t_stat, p_value = ttest_1samp(trend_slopes, 0)
        p_value_one_sided = p_value / 2  # 片側検定のp値

        # 結果を表示
        print(f"Average trend_slope: {average_trend_slope}")
        print(f"Median trend_slope: {median_trend_slope}")
        print(f"Standard deviation of trend_slope: {std_trend_slope}")
        print(f"T-statistic: {t_stat}, P-value (one-sided): {p_value_one_sided}")

        # 優位性の判定
        alpha = 0.05  # 有意水準
        if t_stat > 0 and p_value_one_sided < alpha:
            print("The trend_slope is significantly greater than 0 (reject H0).")
        else:
            print("The trend_slope is not significantly greater than 0 (fail to reject H0).")

        # trend_slopeの分布を可視化
        if trend_analysis:
            trend_slopes = [result['trend_slope'] for result in trend_analysis]

            # ヒストグラムとカーネル密度推定（KDE）のプロット
            plt.figure(figsize=(10, 6))
            sns.histplot(trend_slopes, kde=True, bins=20, color='blue', alpha=0.7, label='Trend Slope Distribution')
            plt.axvline(x=np.mean(trend_slopes), color='red', linestyle='--', label='Mean')
            plt.axvline(x=np.median(trend_slopes), color='green', linestyle='--', label='Median')
            plt.title('Distribution of Trend Slope')
            plt.xlabel('Trend Slope')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("No trend analysis results to visualize.")
    else:
        print("No trend analysis results to calculate statistics.")