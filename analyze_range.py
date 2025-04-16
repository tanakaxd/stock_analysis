import pandas as pd
import os

# データディレクトリのパス
data_dir = './data_utf8/2024'  # 必要に応じて変更

# 平均変動幅を計算するための変数
total_range_sum = 0
total_candles = 0
exceed_threshold_count = 0  # 1%を超える変動幅のカウント
threshold = 0.003  # 1%の変動幅

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
                
                # 変動幅を計算（始値と終値で定義）
                df['range'] = abs((df['close'] - df['open']) / df['open'])  # 終値と始値の変動幅の絶対値
                
                # 有効なデータのみを対象に合計とカウントを更新
                valid_ranges = df['range'].dropna()  # NaNを除外
                total_range_sum += valid_ranges.sum()
                total_candles += len(valid_ranges)
                
                # 1%を超える変動幅のカウント
                exceed_threshold_count += (valid_ranges > threshold).sum()
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

# 平均変動幅と1%を超える割合を計算
if total_candles > 0:
    average_range = total_range_sum / total_candles
    exceed_threshold_ratio = exceed_threshold_count / total_candles
    print(f"全ローソク足の平均変動幅: {average_range:.4f}")
    print(f"{threshold*100}%を超える変動幅の割合: {exceed_threshold_ratio:.2%}")
else:
    print("データが不足しています。")