import pandas as pd
import os

# ./data_utf8ディレクトリにUTF-8形式で保存された日経先物のデータがある。
# P:U列が5分足のデータ。順に[時刻、始値、高値、安値、終値、出来高]となっている。
# 最初の寄り付きのローソク足が陽線だった場合、次も陽線になるかを分析する。
# また、寄り付きが陰線だった場合、2本目も陰線になる確率を求める。

# データディレクトリのパス
data_dir = './data_utf8'  # UTF-8形式で保存したディレクトリ

# カウント変数
total_files = 0
total_first_bullish = 0
second_bullish_after_first_bullish = 0 # 陽線->陽線のカウント
second_bearish_after_first_bullish = 0 # 陽線->陰線のカウント
total_first_bearish = 0
second_bullish_after_first_bearish = 0 # 陰線->陽線のカウント
second_bearish_after_first_bearish = 0 # 陰線->陰線のカウント

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
                
                # 数値型に変換
                df['open'] = pd.to_numeric(df['open'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                
                # 最初のローソク足が陽線または陰線か確認
                if len(df) > 1:
                    first_candle = df.iloc[0]
                    second_candle = df.iloc[1]
                    
                    if first_candle['close'] > first_candle['open']:  # 陽線
                        total_first_bullish += 1
                        if second_candle['close'] > second_candle['open']:  # 2本目も陽線
                            second_bullish_after_first_bullish += 1
                        elif second_candle['close'] < second_candle['open']:  # 2本目が陰線
                            second_bearish_after_first_bullish += 1
                    elif first_candle['close'] < first_candle['open']:  # 陰線
                        total_first_bearish += 1
                        if second_candle['close'] > second_candle['open']:  # 2本目が陽線
                            second_bullish_after_first_bearish += 1
                        elif second_candle['close'] < second_candle['open']:  # 2本目も陰線
                            second_bearish_after_first_bearish += 1
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

# 結果を出力
print(f"分析対象の全日付数: {total_files}")
print(f"寄り付き陽線の回数: {total_first_bullish}")
print(f"陽線->陽線の回数: {second_bullish_after_first_bullish}")
print(f"陽線->陰線の回数: {second_bearish_after_first_bullish}")
print(f"寄り付き陰線の回数: {total_first_bearish}")
print(f"陰線->陽線の回数: {second_bullish_after_first_bearish}")
print(f"陰線->陰線の回数: {second_bearish_after_first_bearish}")

# 確率を計算
if total_first_bullish > 0:
    bullish_probability = second_bullish_after_first_bullish / total_first_bullish
    print(f"寄付陽線→2本目も陽線の確率: {bullish_probability:.2%}")
else:
    print("寄り付き陽線のデータが不足しています。")

if total_first_bearish > 0:
    bearish_probability = second_bearish_after_first_bullish / total_first_bearish
    print(f"寄付陰線→2本目も陰線の確率: {bearish_probability:.2%}")
else:
    print("寄り付き陰線のデータが不足しています。")