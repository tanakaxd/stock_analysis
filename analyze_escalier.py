import pandas as pd
import os

# ./data_utf8ディレクトリに保存された日経先物のデータを分析する。
# 小ローソク足、大ローソク足、小ローソク足で構成されたパターンを「エスカリエ」と定義する。
# 両端ローソク足は陽線陰線関係なく、真ん中が大きい陽線であることが条件。

# データディレクトリのパス
data_dir = './data_utf8/2024/11'  # UTF-8形式で保存したディレクトリ

# カウント変数
total_escalier_patterns = 0
next_candle_up = 0

# 小ローソク足と大ローソク足の閾値
small_candle_threshold = 0.001  # 0.1%以下の変動幅を小ローソク足と定義
large_candle_threshold = 0.003   # 0.3%以上の変動幅を大ローソク足と定義

# ディレクトリ内のCSVファイルを取得
for root, _, files in os.walk(data_dir):
    for file_name in sorted(files):  # 日付順に処理するためソート
        if file_name.endswith('.csv'):
            file_path = os.path.join(root, file_name)
            try:
                # CSVファイルを読み込む
                df = pd.read_csv(file_path, header=[0, 1])
                
                # 必要な列を抽出
                df = df.iloc[:, 15:21]  # P:U列を抽出
                df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
                
                # 数値型に変換
                df['open'] = pd.to_numeric(df['open'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df['high'] = pd.to_numeric(df['high'], errors='coerce')
                df['low'] = pd.to_numeric(df['low'], errors='coerce')
                
                # ローソク足の変動幅を計算、絶対値とする
                df['range'] = abs((df['close'] - df['open']) / df['open'])  # 終値と始値の変動幅の絶対値
                # df['range'] = (df['close'] - df['open']) / df['open'] 

                # dfの最初の１０行に関して、rangeが計算できているかデバグ用に確認
                print(df.head(30))

                # エスカリエパターンを検出
                for i in range(1, len(df) - 2):
                    prev_candle = df.iloc[i - 1]
                    middle_candle = df.iloc[i]
                    next_candle = df.iloc[i + 1]
                    after_next_candle = df.iloc[i + 2]
                    
                    # 条件: 小ローソク足、大ローソク足、小ローソク足
                    if (
                        prev_candle['range'] < small_candle_threshold and
                        middle_candle['range'] > large_candle_threshold and
                        next_candle['range'] < small_candle_threshold and
                        middle_candle['close'] > middle_candle['open']  # 真ん中が陽線
                    ):
                        print(f"Escalier pattern found in {file_name} at index {i}:")
                        total_escalier_patterns += 1
                        
                        # 次のローソク足が上昇しているか確認
                        if after_next_candle['close'] > after_next_candle['open']:
                            next_candle_up += 1
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

# 結果を出力
print(f"エスカリエパターンの総数: {total_escalier_patterns}")
print(f"次のローソク足が上昇した回数: {next_candle_up}")

# 確率を計算
if total_escalier_patterns > 0:
    probability = next_candle_up / total_escalier_patterns
    print(f"次のローソク足が上昇する確率: {probability:.2%}")
else:
    print("エスカリエパターンが見つかりませんでした。")