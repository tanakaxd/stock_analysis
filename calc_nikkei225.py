import pandas as pd
import yfinance as yf
from datetime import datetime
import traceback
from tqdm import tqdm
import random

# IPOデータを読み込む
ipo_csv_file = "ipo_list_2024_2021.csv"
ipo_data = pd.read_csv(ipo_csv_file)
ipo_data = ipo_data.iloc[41:]  # 不要な行を削除

# 公開価格を数値に変換（カンマを削除してfloat型に変換）
ipo_data["public_price"] = ipo_data["public_price"].replace(",", "", regex=True).astype(float)

# 日経225採用銘柄データを読み込む
nikkei225_csv_file = "nikkei225_companies.csv"
nikkei225_data = pd.read_csv(nikkei225_csv_file)

# 日経225のティッカーリストを作成
nikkei225_tickers = nikkei225_data["ticker"].astype(str) + ".T"

# 結果を格納するリスト
comparison_results = []

# 各IPO銘柄のデータを取得
for index, row in tqdm(ipo_data.iterrows(), total=len(ipo_data), desc="Processing IPO data"):
    ipo_ticker = row["ticker"]
    ipo_listing_date = datetime.strptime(row["listing_date"].strip(), "%Y/%m/%d")

    # ランダムに日経225採用銘柄を選択
    comparison_ticker = random.choice(nikkei225_tickers)

    try:
        # 比較対象企業の株価データを取得
        stock = yf.Ticker(comparison_ticker)
        history = stock.history(start=ipo_listing_date, end=datetime.now(), auto_adjust=False, interval="1d")

        # 銘柄名を取得
        company_name = stock.info.get("longName", "Unknown")  # 銘柄名が取得できない場合は "Unknown" を設定

        if history.empty:
            print(f"No data available for {comparison_ticker}.")
            continue

        # 初値（上場日の始値）
        first_open_price = history.iloc[0]["Open"] if not history.empty else None

        # 上場から180営業日後までの終値を取得
        daily_prices = history["Close"].iloc[:180]  # 最初の180営業日分の終値を取得

        # データ構造を指定された形式に変換
        result = {
            "comparison_ticker": comparison_ticker,
            "ipo_ticker": ipo_ticker,
            "company_name": company_name,  # 銘柄名を追加
            "listing_date": ipo_listing_date.strftime("%Y-%m-%d"),
        }

        # 各営業日の終値をprice_1_day, price_2_day, ..., price_180_dayとして追加
        for i, price in enumerate(daily_prices):
            result[f"price_{i + 1}_day"] = price

        # 現在価格（最新の終値）
        result["current_price"] = history["Close"].iloc[-1] if not history.empty else None

        # 結果をリストに追加
        comparison_results.append(result)

    except Exception as e:
        print(f"Failed to process {comparison_ticker}: {e}")
        traceback.print_exc()

# 結果をデータフレームに変換
comparison_results_df = pd.DataFrame(comparison_results)

# 結果を表示
print(comparison_results_df)

# 結果をCSVに保存
comparison_results_df.to_csv("comparison_data.csv", index=False)