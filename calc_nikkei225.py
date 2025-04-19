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

        if history.empty:
            print(f"No data available for {comparison_ticker}.")
            continue

        # 初値（上場日の始値）
        first_open_price = history.iloc[0]["Open"] if not history.empty else None

        # 上場10営業日後の終値
        price_10_days = history.iloc[9]["Close"] if len(history) > 9 else None

        # 上場30営業日後の終値
        price_30_days = history.iloc[29]["Close"] if len(history) > 29 else None

        # 上場60営業日後の終値
        price_60_days = history.iloc[59]["Close"] if len(history) > 59 else None

        # 上場90営業日後の終値
        price_90_days = history.iloc[89]["Close"] if len(history) > 89 else None

        # 上場180営業日後の終値
        price_180_days = history.iloc[179]["Close"] if len(history) > 179 else None

        # 現在価格（最新の終値）
        current_price = history["Close"].iloc[-1] if not history.empty else None

        # 結果をリストに追加
        comparison_results.append({
            "comparison_ticker": comparison_ticker,
            "ipo_ticker": ipo_ticker,
            "first_open_price": first_open_price,
            "price_10_days": price_10_days,
            "price_30_days": price_30_days,
            "price_60_days": price_60_days,
            "price_90_days": price_90_days,
            "price_180_days": price_180_days,
            "current_price": current_price
        })

    except Exception as e:
        print(f"Failed to process {comparison_ticker}: {e}")
        traceback.print_exc()

# 結果をデータフレームに変換
comparison_results_df = pd.DataFrame(comparison_results)

# 結果を表示
print(comparison_results_df)

# 結果をCSVに保存
comparison_results_df.to_csv("comparison_data.csv", index=False)