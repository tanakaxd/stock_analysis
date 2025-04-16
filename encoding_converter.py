import os
import pandas as pd

# ディレクトリのパス
input_dir = "./data"
output_dir = "./data_utf8"

# UTF-8形式で保存するディレクトリを作成
os.makedirs(output_dir, exist_ok=True)

# ディレクトリ内のCSVファイルを変換
for file_name in os.listdir(input_dir):
    if file_name.endswith(".csv"):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        try:
            # Shift-JIS形式で読み込む
            df = pd.read_csv(input_path, encoding="shift-jis")
            
            # UTF-8形式で保存する
            df.to_csv(output_path, index=False, encoding="utf-8")
            print(f"Converted and saved: {output_path}")
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")