import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd

# JPXのIPOリストページURL
url = "https://www.jpx.co.jp/listing/stocks/new/00-archives-01.html"

# JPXのウェブページをスクレイピング
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# IPO銘柄の上場日、会社名、銘柄コード、市場区分、公開価格を取得
# 銘柄コードを抽出（<span id="302A">のような形式を満たす要素）
ipo_list = []
for span in soup.find_all("span", id=True):  # id属性を持つ<span>タグを検索
    ticker = span["id"] + ".T"  # 銘柄コードに".T"を追加
    ipo_list.append({"ticker": ticker})

# IPOリストを表示
print("IPOリスト:", ipo_list)
ipo_list = ipo_list[:5]  # 最初の10銘柄を表示（デバッグ用）
# IPOリストを表示（デバッグ用）
print("IPOリスト:", ipo_list)

# 分析期間を指定
start_date = "2024-01-01"
end_date = "2024-12-31"

# IPOデータを取得
ipo_results = []
for ipo in ipo_list:
    ticker = ipo["ticker"]
    print(f"Processing {ticker}...")

    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=False)
        df.reset_index(inplace=True)

        # multi-indexの場合、列名をフラット化
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 列名を小文字に変換
        df.columns = df.columns.str.lower()

        print(f"Columns after flattening: {df.columns.tolist()}")  # デバッグ用出力
        print(f"Data for {ticker}:\n", df.head())  # デバッグ用出力

        if not df.empty:
            if "open" in df.columns:
                first_open_price = df.at[0, "open"]  # 小文字の"open"を使用
                ipo_results.append({
                    "ticker": ticker,
                    "first_open_price": first_open_price
                })
            else:
                print(f"'open' column not found for {ticker}")
        else:
            print(f"No data found for {ticker}")
    except Exception as e:
        print(f"Failed to process {ticker}: {e}")

# IPO結果を表示
if not ipo_results:
    print("No IPO data found.")
print("ipo_results:", ipo_results)

# 結果をデータフレームに変換
ipo_results_df = pd.DataFrame(ipo_results)

# 結果を表示
print(ipo_results_df)

# 結果をCSVに保存
ipo_results_df.to_csv("ipo_results_2024.csv", index=False)