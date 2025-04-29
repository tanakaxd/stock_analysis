import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv("nikkei_combined_5min_cleaned.csv")

# datetime列をdatetime型に変換（必要なら）
df["datetime"] = pd.to_datetime(df["datetime"])

# 重複しているdatetimeの行を抽出
duplicate_rows = df[df.duplicated(subset=["datetime"], keep=False)]

# 重複している日時を表示
print("重複しているdatetime:")
print(duplicate_rows)
