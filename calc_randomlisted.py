import pandas as pd
import yfinance as yf
from datetime import datetime
import traceback
from tqdm import tqdm
import random

# 全上場企業リストを読み込む
all_companies_csv = "all_listed_companies.csv"
all_companies_data = pd.read_csv(all_companies_csv)

# IPOデータを読み込む（IPO銘柄数に合わせて基準群を選出する場合に使用）
ipo_csv_file = "ipo_list_2024_2021.csv"
ipo_data = pd.read_csv(ipo_csv_file)
ipo_data = ipo_data.iloc[41:]  # 不要な行を削除

# ランダムに銘柄を選出（IPO銘柄数と同じ数を選ぶ）
random_tickers = random.sample(all_companies_data["ticker"].astype(str).tolist(), len(ipo_data))

# 結果を格納するリスト
comparison_results = []

# 各ランダム選出銘柄のデータを取得
for ticker in tqdm(random_tickers, desc="Processing Randomly Selected Companies"):
    try:
        # 株価データを取得
        stock = yf.Ticker(ticker + ".T")  # 日本株の場合、".T"を付ける
        history = stock.history(period="1y", auto_adjust=False, interval="1d")

        # 銘柄名を取得
        company_name = stock.info.get("longName", "Unknown")  # 銘柄名が取得できない場合は "Unknown" を設定

        if history.empty:
            print(f"No data available for {ticker}.")
            continue

        # 初値（最初の営業日の始値）
        first_open_price = history.iloc[0]["Open"] if not history.empty else None

        # 上場から180営業日後までの終値を取得
        daily_prices = history["Close"].iloc[:180]  # 最初の180営業日分の終値を取得

        # データ構造を指定された形式に変換
        result = {
            "ticker": ticker,
            "company_name": company_name,
            "first_open_price": first_open_price,
        }

        # 各営業日の終値をprice_1_day, price_2_day, ..., price_180_dayとして追加
        for i, price in enumerate(daily_prices):
            result[f"price_{i + 1}_day"] = price

        # 現在価格（最新の終値）
        result["current_price"] = history["Close"].iloc[-1] if not history.empty else None

        # 結果をリストに追加
        comparison_results.append(result)

    except Exception as e:
        print(f"Failed to process {ticker}: {e}")
        traceback.print_exc()

# 結果をデータフレームに変換
comparison_results_df = pd.DataFrame(comparison_results)

# 結果を表示
print(comparison_results_df)

# 結果をCSVに保存
comparison_results_df.to_csv("comparison_data_random.csv", index=False)