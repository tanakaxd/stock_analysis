import requests
from bs4 import BeautifulSoup
import pandas as pd

# JPXのIPOリストページURL
url = "https://www.jpx.co.jp/listing/stocks/new/00-archives-01.html"

# JPXのウェブページをスクレイピング
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# IPO情報を格納するリスト
ipo_list = []

# テーブルの行を取得
rows = soup.find_all("tr")

# 各行から必要な情報を抽出
for i in range(0, len(rows), 2):  # 2行で1つのIPO情報を構成
    try:
        # 上場日
        listing_date_cell = rows[i].find("td", class_=["a-center tb-color001 w-space", "a-center tb-color002 w-space"])
        listing_date = listing_date_cell.text.strip().split("\n")[0] if listing_date_cell else None

        # 会社名
        company_name_cell = rows[i].find("td", class_=["a-left tb-color001 issuename-word-break", "a-left tb-color002 issuename-word-break"])
        company_name = company_name_cell.text.strip().split("\n")[0] if company_name_cell else None

        # 銘柄コード
        ticker_cell = rows[i].find("span", id=True)
        ticker = ticker_cell["id"] + ".T" if ticker_cell else None

        # 市場区分
        market_division_cell = rows[i + 1].find("td", class_=["a-center tb-color001", "a-center tb-color002"])
        market_division = market_division_cell.text.strip() if market_division_cell else None

        # 公開価格
        public_price_cells = rows[i + 1].find_all("td", class_=["a-right w-space tb-color001", "a-right w-space tb-color002"])
        public_price = public_price_cells[-2].text.strip() if len(public_price_cells) >= 2 else None

        # データをリストに追加（すべての値が取得できた場合のみ）
        if listing_date and company_name and ticker and market_division and public_price:
            ipo_list.append({
                "listing_date": listing_date,
                "company_name": company_name,
                "ticker": ticker,
                "market_division": market_division,
                "public_price": public_price
            })
        else:
            print(f"Skipping row {i} due to missing data.")
    except Exception as e:
        print(f"Error processing row {i}: {e}")

# 結果をデータフレームに変換
ipo_df = pd.DataFrame(ipo_list)

# 結果を表示
print(ipo_df)

# 結果をCSVに保存
ipo_df.to_csv("ipo_list_2024.csv", index=False)