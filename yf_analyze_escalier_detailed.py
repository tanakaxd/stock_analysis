import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. 日経先物の2025年4月の全5分足データを取得
def fetch_nikkei_futures_data():
    # ticker = "NIY=F"  # 日経先物のティッカー
    ticker = "^N225"  # 日経平均のティッカー
    df = yf.download(ticker, start="2025-04-01", end="2025-04-18", interval="5m")
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
    data = data.copy()  # データをコピーして元のデータを変更しないようにする
    candles_ahead_count = 5  # 何本先のローソク足までを対象として分析するか

    for pattern_time in patterns:
        # パターン後のデータを取得
        pattern_index = data.index.get_loc(pattern_time)
        future_data = data.iloc[pattern_index:pattern_index + candles_ahead_count]

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

    # trend_slopeの平均値を計算
    if trend_analysis:
        trend_slopes = [result['trend_slope'] for result in trend_analysis]
        average_trend_slope = sum(trend_slopes) / len(trend_slopes)
        print(f"Average trend_slope: {average_trend_slope}")
    else:
        print("No trend analysis results to calculate average trend_slope.")