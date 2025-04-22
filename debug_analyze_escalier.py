import pandas as pd
import os

# データディレクトリのパス
data_dir = './data_utf8'  # UTF-8形式で保存したディレクトリ

# カウント変数
total_candles = 0
up_candles = 0
down_candles = 0
no_change_candles = 0

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
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                
                # 各ローソク足の状態を判定（終値と次のローソク足の終値を比較）
                for i in range(len(df) - 1):  # 最後のローソク足は次がないため除外
                    total_candles += 1
                    if df.iloc[i + 1]['close'] > df.iloc[i]['close']:
                        up_candles += 1
                    elif df.iloc[i + 1]['close'] < df.iloc[i]['close']:
                        down_candles += 1
                    else:
                        no_change_candles += 1
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

# 確率を計算
if total_candles > 0:
    up_probability = up_candles / total_candles
    down_probability = down_candles / total_candles
    no_change_probability = no_change_candles / total_candles

    # 結果を出力
    print(f"総ローソク足数: {total_candles}")
    print(f"上昇するローソク足の数: {up_candles} ({up_probability:.2%})")
    print(f"下降するローソク足の数: {down_candles} ({down_probability:.2%})")
    print(f"変化しないローソク足の数: {no_change_candles} ({no_change_probability:.2%})")
else:
    print("ローソク足データが見つかりませんでした。")