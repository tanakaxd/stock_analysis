import requests
from bs4 import BeautifulSoup
import csv

# 日経225構成銘柄のURL
url = "https://indexes.nikkei.co.jp/nkave/index/component?idx=nk225"

# HTTPリクエストを送信してHTMLを取得
response = requests.get(url)
response.encoding = response.apparent_encoding  # 文字コードを自動検出して設定
html = response.text

# BeautifulSoupでHTMLを解析
soup = BeautifulSoup(html, "html.parser")

# 銘柄コードと銘柄名を取得
data = []
for tr in soup.find_all("tr"):
    td = tr.find("td")  # 最初の<td>要素を取得（銘柄コード）
    a = tr.find("td").find_next("td").find("a") if td else None  # 次の<td>内の<a>タグ（銘柄名）
    if td and a:
        ticker = td.text.strip()
        company_name = a.text.strip()
        data.append({"ticker": ticker, "company_name": company_name})

# 結果をCSVファイルに保存
csv_file = "nikkei225_companies.csv"
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["ticker", "company_name"])
    writer.writeheader()
    writer.writerows(data)

print(f"Saved {len(data)} entries to {csv_file}")