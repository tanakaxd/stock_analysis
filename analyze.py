import pandas as pd
import os

# ./dataディレクトリにcsv形式で日経先物のデータが保存されている。
# P:U列が5分足のデータ。順に[時刻、始値、高値、安値、終値、出来高]となっている。
# 最初の寄り付きのローソク足が陽線だった場合、次も陽線になるか。要は「寄付陽線→2本目も陽線」となっている確率を知りたい。
# データディレクトリのパス
data_dir = './data_utf8'  # UTF-8形式で保存したディレクトリ

# CSVファイルを取得
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# 寄付陽線→2本目も陽線のカウント
total_first_bullish = 0
second_bullish_after_first = 0

for file in csv_files:
    # CSVファイルを読み込む
    df = pd.read_csv(os.path.join(data_dir, file))
    
    # 必要な列を抽出
    df = df.iloc[:, 15:21]  # P:U列を抽出
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    
    # 最初のローソク足が陽線か確認
    if len(df) > 1:
        first_candle = df.iloc[0]
        second_candle = df.iloc[1]
        
        if first_candle['close'] > first_candle['open']:  # 陽線
            total_first_bullish += 1
            if second_candle['close'] > second_candle['open']:  # 2本目も陽線
                second_bullish_after_first += 1

# 確率を計算
if total_first_bullish > 0:
    probability = second_bullish_after_first / total_first_bullish
    print(f"寄付陽線→2本目も陽線の確率: {probability:.2%}")
else:
    print("データが不足しています。")