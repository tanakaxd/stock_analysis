import pandas as pd
import os

# ./data_utf8/2024/04ディレクトリにUTF-8形式で保存された日経先物のデータがある。
# P:U列が5分足のデータ。順に[時刻、始値、高値、安値、終値、出来高]となっている。
# 最初の寄り付きのローソク足が陽線だった場合、次も陽線になるかを分析する。

# データディレクトリのパス（2024年4月のみ）
data_dir = './data_utf8/2024/04'

# 寄付陽線→2本目も陽線のカウント
total_files = 0
total_first_bullish = 0
second_bullish_after_first = 0

# ディレクトリ内のCSVファイルを取得
for root, _, files in os.walk(data_dir):
    for file_name in files:
        if file_name.endswith('.csv'):
            total_files += 1  # 分析対象の全日付数をカウント
            file_path = os.path.join(root, file_name)
            try:
                # CSVファイルを読み込む
                df = pd.read_csv(file_path, header=[0, 1])
                
                # 必要な列を抽出
                df = df.iloc[:, 15:21]  # P:U列を抽出
                df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
                
                # 最初のローソク足が陽線か確認
                if len(df) > 1:
                    first_candle = df.iloc[0]
                    second_candle = df.iloc[1]
                    
                    # デバッグ用出力
                    print(f"Processing file: {file_name}")
                    print(f"First candle: {first_candle}")
                    
                    if first_candle['close'] > first_candle['open']:  # 陽線
                        total_first_bullish += 1
                        if second_candle['close'] > second_candle['open']:  # 2本目も陽線
                            second_bullish_after_first += 1
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

# 結果を出力
print(f"分析対象の全日付数: {total_files}")
print(f"寄り付き陽線の回数: {total_first_bullish}")
print(f"2本目も陽線の回数: {second_bullish_after_first}")

# 確率を計算
if total_first_bullish > 0:
    probability = second_bullish_after_first / total_first_bullish
    print(f"寄付陽線→2本目も陽線の確率: {probability:.2%}")
else:
    print("データが不足しています。")