import yfinance as yf
import pandas as pd
import time

# 2024年のIPO銘柄リスト（例: 銘柄コードと公募価格）
ipo_list = [
    {"ticker": "6524.T", "ipo_price": 1500},  # 例: 銘柄コードと公募価格
    {"ticker": "7373.T", "ipo_price": 1200},
    {"ticker": "9250.T", "ipo_price": 1800},
]

# 分析期間を指定（IPO初日を含む期間）
start_date = "2024-01-01"
end_date = "2024-12-31"

# 結果を格納するリスト
ipo_results = []

# 各IPO銘柄を処理
for ipo in ipo_list:
    ticker = ipo["ticker"]
    ipo_price = ipo["ipo_price"]
    print(f"Processing {ticker}...")

    # リトライ機能付きデータ取得
    max_retries = 5
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=False)
            df.reset_index(inplace=True)
            break  # 成功したらループを抜ける
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5)  # 5秒待機して再試行
    else:
        print(f"データの取得に失敗しました: {ticker}")
        continue

    # 公開価格（初日の始値）を取得
    if not df.empty:
        first_open_price = df.iloc[0]["Open"]  # 初日の始値
        ipo_results.append({
            "ticker": ticker,
            "ipo_price": ipo_price,
            "first_open_price": first_open_price
        })
    else:
        print(f"データが空です: {ticker}")

# 結果をデータフレームに変換
ipo_results_df = pd.DataFrame(ipo_results)

# 結果を表示
print(ipo_results_df)

# 結果をCSVに保存（必要に応じて）
ipo_results_df.to_csv("ipo_results_2024.csv", index=False)