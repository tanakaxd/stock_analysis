import os
import pandas as pd

# 入力ディレクトリと出力ディレクトリのパス
input_dir = "./data"
output_dir = "./data_utf8"

# UTF-8形式で保存するディレクトリを作成
os.makedirs(output_dir, exist_ok=True)

# ディレクトリ内のCSVファイルを再帰的に変換
for root, _, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith(".csv"):
            # 入力ファイルのパス
            input_path = os.path.join(root, file_name)
            
            # 出力ファイルのパス（ディレクトリ構造を維持）
            relative_path = os.path.relpath(root, input_dir)
            output_sub_dir = os.path.join(output_dir, relative_path)
            os.makedirs(output_sub_dir, exist_ok=True)
            output_path = os.path.join(output_sub_dir, file_name)
            
            try:
                # Shift-JIS形式で読み込む
                df = pd.read_csv(input_path, encoding="shift-jis")
                
                # UTF-8形式で保存する
                df.to_csv(output_path, index=False, encoding="utf-8")
                print(f"Converted and saved: {output_path}")
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")