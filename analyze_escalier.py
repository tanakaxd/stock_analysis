import pandas as pd
import os

# ./data_utf8ディレクトリに保存された日経先物のデータを分析する。
# 小ローソク足、大ローソク足、小ローソク足で構成されたパターンを「エスカリエ」と定義する。
# 両端ローソク足は陽線陰線関係なく、真ん中が大きい陽線であることが条件。

# データディレクトリのパス
data_dir = './data_utf8/2024'  # UTF-8形式で保存したディレクトリ

# カウント変数
total_escalier_patterns = 0
next_candle_up = 0
fifth_candle_up = 0
tenth_candle_up = 0

# 小ローソク足と大ローソク足の閾値
# 全ローソク足の平均変動幅: 0.0007 => 0.07%
# 1%を超える変動幅の割合: 0.05%
# 0.8%を超える変動幅の割合: 0.13%
# 0.5%を超える変動幅の割合: 0.50%
# 0.3%を超える変動幅の割合: 1.71%
small_candle_threshold = 0.0005  # 0.05%以下の変動幅を小ローソク足と定義
large_candle_threshold = 0.001   # 0.1%以上の変動幅を大ローソク足と定義

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
                # df['range'] = (df['close'] - df['open']) / df['open'] 

                # dfの最初の１０行に関して、rangeが計算できているかデバグ用に確認
                # print(df.head(10))
                # print(df.tail(10))

                # エスカリエパターンを検出
                for i in range(1, len(df) - 10):  # 10番目のローソク足まで確認するために範囲を調整
                    prev_candle = df.iloc[i - 1]
                    middle_candle = df.iloc[i]
                    next_candle = df.iloc[i + 1]
                    after_next_candle = df.iloc[i + 2]
                    fifth_candle = df.iloc[i + 5] if i + 5 < len(df) else None  # 5番目のローソク足
                    tenth_candle = df.iloc[i + 10] if i + 10 < len(df) else None  # 10番目のローソク足
                    
                    # 条件: 小ローソク足、大ローソク足、小ローソク足
                    if (
                        prev_candle['range'] < small_candle_threshold and
                        middle_candle['range'] > large_candle_threshold and
                        next_candle['range'] < small_candle_threshold and
                        middle_candle['close'] > middle_candle['open']  # 真ん中が陽線
                    ):
                        print(f"Escalier pattern found in {file_name} at index {i}:")
                        print(f"  Previous Candle: {prev_candle['range']:.3%}")
                        print(f"  Middle Candle: {middle_candle['range']:.3%}")
                        print(f"  Next Candle: {next_candle['range']:.3%}")
                        print(f"  After Next Candle: {after_next_candle['close'] - after_next_candle['open']}")
                        total_escalier_patterns += 1
                        
                        # 次のローソク足が上昇しているか確認
                        if after_next_candle['close'] > after_next_candle['open']:
                            next_candle_up += 1
                        
                        # 5番目のローソク足がmiddle_candleの終値より上昇しているか確認
                        if fifth_candle is not None:
                            fifth_candle_delta = fifth_candle['close'] - middle_candle['close']
                            if fifth_candle_delta > 0:
                                # 5番目のローソク足がmiddle_candleの終値より上昇している場合
                                fifth_candle_up += 1
                            print(f"  5th Candle delta: {fifth_candle_delta}")
                        else:
                            print("  5th Candle: Not available")
                        
                        # 10番目のローソク足がmiddle_candleの終値より上昇しているか確認
                        if tenth_candle is not None:
                            tenth_candle_delta = tenth_candle['close'] - middle_candle['close']
                            if tenth_candle_delta > 0:
                                # 10番目のローソク足がmiddle_candleの終値より上昇している場合
                                tenth_candle_up += 1
                            print(f"  10th Candle delta: {tenth_candle_delta}")
                        else:
                            print("  10th Candle: Not available")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

# 結果を出力
print(f"エスカリエパターンの総数: {total_escalier_patterns}")
print(f"次のローソク足が上昇した回数: {next_candle_up}")
print(f"5番目のローソク足がmiddle_candleの終値より上昇した回数: {fifth_candle_up}")
print(f"10番目のローソク足がmiddle_candleの終値より上昇した回数: {tenth_candle_up}")

# 確率を計算
if total_escalier_patterns > 0:
    probability_next = next_candle_up / total_escalier_patterns
    probability_fifth = fifth_candle_up / total_escalier_patterns
    probability_tenth = tenth_candle_up / total_escalier_patterns
    print(f"次のローソク足が上昇する確率: {probability_next:.2%}")
    print(f"5番目のローソク足がmiddle_candleの終値より上昇する確率: {probability_fifth:.2%}")
    print(f"10番目のローソク足がmiddle_candleの終値より上昇する確率: {probability_tenth:.2%}")
else:
    print("エスカリエパターンが見つかりませんでした。")