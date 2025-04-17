import yfinance as yf
import pandas as pd

# 銘柄と期間を指定
ticker = "AAPL"  # 例: Appleの株価データ
start_date = "2024-01-01"
end_date = "2024-12-31"

# データを取得
df = yf.download(ticker, start=start_date, end=end_date, interval="1d")  # 日足データ
df.reset_index(inplace=True)
# 必要な列をリネーム
df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

# print(df.head(5))  # データの最初の5行を表示

# カウント変数
total_escalier_patterns = 0
next_candle_up = 0
fifth_candle_up = 0
tenth_candle_up = 0

# 小ローソク足と大ローソク足の閾値
small_candle_threshold = 0.005  # 0.5%以下の変動幅を小ローソク足と定義
large_candle_threshold = 0.02   # 2%以上の変動幅を大ローソク足と定義

# ローソク足の変動幅を計算
df['range'] = abs((df['close'] - df['open']) / df['open'])  # 終値と始値の変動幅の絶対値

print(df.head(5))  # データの最初の5行を表示


# エスカリエパターンを検出
for i in range(1, len(df) - 10):  # 10番目のローソク足まで確認するために範囲を調整
    prev_candle_range = df.iloc[i - 1]['range']
    middle_candle_range = df.iloc[i]['range']
    next_candle_range = df.iloc[i + 1]['range']
    middle_candle_close = df.iloc[i]['close']
    middle_candle_open = df.iloc[i]['open']
    after_next_candle_close = df.iloc[i + 2]['close']
    after_next_candle_open = df.iloc[i + 2]['open']
    fifth_candle_close = df.iloc[i + 5]['close'] if i + 5 < len(df) else None
    tenth_candle_close = df.iloc[i + 10]['close'] if i + 10 < len(df) else None
    
    # 条件: 小ローソク足、大ローソク足、小ローソク足
    if (
        prev_candle_range < small_candle_threshold and
        middle_candle_range > large_candle_threshold and
        next_candle_range < small_candle_threshold and
        middle_candle_close > middle_candle_open  # 真ん中が陽線
    ):
        total_escalier_patterns += 1
        
        # 次のローソク足が上昇しているか確認
        if after_next_candle_close > after_next_candle_open:
            next_candle_up += 1
        
        # 5番目のローソク足がmiddle_candleの終値より上昇しているか確認
        if fifth_candle_close is not None and fifth_candle_close > middle_candle_close:
            fifth_candle_up += 1
        
        # 10番目のローソク足がmiddle_candleの終値より上昇しているか確認
        if tenth_candle_close is not None and tenth_candle_close > middle_candle_close:
            tenth_candle_up += 1

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