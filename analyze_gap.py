import pandas as pd
import os

# ./data_utf8/2024/04ディレクトリにUTF-8形式で保存された日経先物のデータがある。
# P:U列が5分足のデータ。列は順に[時刻、始値、高値、安値、終値、出来高]となっている。
# 始値のデータがヘッダーを抜いた一番上の行にあり、終値のデータは79行目にある。5分足のデータでは15:15が終値。

# データディレクトリのパス
data_dir = './data_utf8'

# カウント変数
total_files = 0
gap_up_count = 0
gap_down_count = 0
gap_up_higher_close = 0
gap_up_lower_close = 0
gap_down_higher_close = 0
gap_down_lower_close = 0

# 前日の終値を保持する変数
previous_close = None

# ギャップアップ・ギャップダウンの閾値（3%）
gap_threshold = 0.02

# ディレクトリ内のCSVファイルを取得
for root, _, files in os.walk(data_dir):
    for file_name in sorted(files):  # 日付順に処理するためソート
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
                
                # 始値と終値を取得
                current_open = df.iloc[0]['open']  # 一番上の行の始値
                current_close = df.iloc[78]['close']  # 79行目の終値
                
                # 前日の終値が存在する場合、ギャップを計算
                if previous_close is not None and not pd.isna(previous_close):
                    
                    # ギャップアップの判定（3%以上上昇）
                    if (current_open - previous_close) / previous_close > gap_threshold:
                        # ギャップアップが発生した日付と乖離度を表示
                        print(f"Gap Up on {file_name}: Previous Close: {previous_close}, Current Open: {current_open}, Gap: {(current_open - previous_close) / previous_close:.2%}")
                        print(f"Current Open: {current_open} => Current Close: {current_close}")
                        gap_up_count += 1
                        if current_close > current_open:  # 終値が始値より高い
                            gap_up_higher_close += 1
                        elif current_close < current_open:  # 終値が始値より低い
                            gap_up_lower_close += 1
                        # else:  # 終値が始値と同じ

                    # ギャップダウンの判定（3%以上下落）
                    elif (previous_close - current_open) / previous_close > gap_threshold:
                        # ギャップダウンが発生した日付を表示
                        print(f"Gap Down on {file_name}: Previous Close: {previous_close}, Current Open: {current_open}, Gap: -{(previous_close - current_open) / previous_close:.2%}")
                        print(f"Current Open: {current_open} => Current Close: {current_close}")
                        gap_down_count += 1
                        if current_close > current_open:  # 終値が始値より高い
                            gap_down_higher_close += 1
                        elif current_close < current_open:  # 終値が始値より低い
                            gap_down_lower_close += 1
                        # else:  # 終値が始値と同じ
                
                # 現在の終値を次の日の前日の終値として保存
                previous_close = current_close
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

# 結果を出力
print(f"分析対象の全日付数: {total_files}")
print(f"ギャップアップの回数: {gap_up_count}")
print(f"ギャップアップ→終値が始値より高い: {gap_up_higher_close}")
print(f"ギャップアップ→終値が始値より低い: {gap_up_lower_close}")
print(f"ギャップダウンの回数: {gap_down_count}")
print(f"ギャップダウン→終値が始値より高い: {gap_down_higher_close}")
print(f"ギャップダウン→終値が始値より低い: {gap_down_lower_close}")

# 確率を計算
if gap_up_count > 0:
    gap_up_higher_probability = gap_up_higher_close / gap_up_count
    print(f"ギャップアップ→終値が始値より高い確率: {gap_up_higher_probability:.2%}")
    gap_up_lower_probability = gap_up_lower_close / gap_up_count
    print(f"ギャップアップ→終値が始値より低い確率: {gap_up_lower_probability:.2%}")

if gap_down_count > 0:
    gap_down_higher_probability = gap_down_higher_close / gap_down_count
    print(f"ギャップダウン→終値が始値より高い確率: {gap_down_higher_probability:.2%}")
    gap_down_lower_probability = gap_down_lower_close / gap_down_count
    print(f"ギャップダウン→終値が始値より低い確率: {gap_down_lower_probability:.2%}")