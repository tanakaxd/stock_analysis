import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

# データディレクトリのパス
data_dir = './data_utf8/2024/10'

# カウント変数
total_escalier_patterns = 0
next_candle_up = 0
fifth_candle_up = 0
tenth_candle_up = 0

# 小ローソク足と大ローソク足の閾値
small_candle_threshold = 0.0005  # 0.05%以下の変動幅を小ローソク足と定義
large_candle_threshold = 0.001   # 0.1%以上の変動幅を大ローソク足と定義

# トレンド判別用の関数
def calculate_trend_indicators(df):
    """指定された範囲のデータでトレンド指標を計算"""
    # スライスを明示的にコピー
    trend_data = df.copy()

    # 移動平均線
    trend_data['MA_5'] = trend_data['close'].rolling(window=5).mean()
    trend_data['MA_20'] = trend_data['close'].rolling(window=20).mean()

    # RSI
    delta = trend_data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    trend_data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    trend_data['EMA_12'] = trend_data['close'].ewm(span=12, adjust=False).mean()
    trend_data['EMA_26'] = trend_data['close'].ewm(span=26, adjust=False).mean()
    trend_data['MACD'] = trend_data['EMA_12'] - trend_data['EMA_26']
    trend_data['Signal'] = trend_data['MACD'].ewm(span=9, adjust=False).mean()

    # ボリンジャーバンド
    trend_data['BB_upper'] = trend_data['close'].rolling(window=20).mean() + 2 * trend_data['close'].rolling(window=20).std()
    trend_data['BB_lower'] = trend_data['close'].rolling(window=20).mean() - 2 * trend_data['close'].rolling(window=20).std()

    # 線形回帰の傾き
    slopes = []
    window = 20
    for i in range(len(trend_data) - window + 1):
        y = trend_data['close'].iloc[i:i+window].values
        x = np.arange(window).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        slopes.append(model.coef_[0])
    trend_data['Slope'] = pd.Series(slopes, index=trend_data.index[window - 1:])

    return trend_data

# ディレクトリ内のCSVファイルを取得
for root, _, files in os.walk(data_dir):
    for file_name in sorted(files):  # 日付順に処理するためソート
        if file_name.endswith('.csv'):
            file_path = os.path.join(root, file_name)
            try:
                # CSVファイルを読み込む
                df = pd.read_csv(file_path, header=[0, 1])
                
                # 必要な行と列を抽出。行は0-78行目、列はP:U列を抽出
                df = df.iloc[:79, 15:21]  # P:U列を抽出
                df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
                
                # 数値型に変換
                df['open'] = pd.to_numeric(df['open'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df['high'] = pd.to_numeric(df['high'], errors='coerce')
                df['low'] = pd.to_numeric(df['low'], errors='coerce')
                
                # ローソク足の変動幅を計算、絶対値とする
                df['range'] = abs((df['close'] - df['open']) / df['open'])  # 終値と始値の変動幅の絶対値

                # エスカリエパターンを検出
                for i in range(1, len(df) - 2):  #
                    prev_candle = df.iloc[i - 1]
                    middle_candle = df.iloc[i]
                    next_candle = df.iloc[i + 1]
                    
                    # 条件: 小ローソク足、大ローソク足、小ローソク足
                    if (
                        prev_candle['range'] < small_candle_threshold and
                        middle_candle['range'] > large_candle_threshold and
                        next_candle['range'] < small_candle_threshold and
                        middle_candle['close'] > middle_candle['open']  # 真ん中が陽線
                    ):
                        print(f"Escalier pattern found in {file_name} at index {i}:")
                        total_escalier_patterns += 1
                        
                        # トレンド指標を計算
                        trend_data = calculate_trend_indicators(df)
                        print(trend_data[['MA_5', 'MA_20', 'RSI', 'MACD', 'Signal', 'BB_upper', 'BB_lower', 'Slope']].iloc[i:i + 5])

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

# 結果を出力
print(f"エスカリエパターンの総数: {total_escalier_patterns}")