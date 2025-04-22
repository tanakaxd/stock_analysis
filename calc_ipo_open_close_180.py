import pandas as pd
import yfinance as yf
from datetime import datetime
import traceback
from tqdm import tqdm  # tqdmライブラリをインポート

# CSVファイルを読み込む
csv_file = "ipo_list_2024_2021.csv"
ipo_data = pd.read_csv(csv_file)
# 取得したデータの最初の45行をinvalidとして削除
ipo_data = ipo_data.iloc[41:]

# 公開価格を数値に変換（カンマを削除してfloat型に変換）
ipo_data["public_price"] = ipo_data["public_price"].replace(",", "", regex=True).astype(float)

# 結果を格納するリスト
results = []

# 各銘柄のデータを取得
for index, row in tqdm(ipo_data.iterrows(), total=len(ipo_data), desc="Processing IPO data"):
    ticker = row["ticker"]  # 銘柄コード
    public_price = row["public_price"]
    listing_date = row["listing_date"].strip()  # 上場日
    listing_date = datetime.strptime(listing_date, "%Y/%m/%d")  # 日付形式に変換

    try:
        # 株価データを取得
        stock = yf.Ticker(ticker)
        history = stock.history(start=listing_date, end=datetime.now(), auto_adjust=False, interval="1d")

        if history.empty:
            print(f"No data available for {ticker}.")
            continue

        # 初値（上場日の始値）
        first_open_price = history.iloc[0]["Open"]  # 最初の行の始値

        # 上場から180営業日後までの始値と終値を取得
        daily_open_prices = history["Open"].iloc[:180]  # 最初の180営業日分の始値を取得
        daily_close_prices = history["Close"].iloc[:180]  # 最初の180営業日分の終値を取得

        # データ構造を指定された形式に変換
        result = {
            "ticker": ticker,
            "company_name": row["company_name"],
            "public_price": public_price,
            "first_open_price": first_open_price
        }

        # 各営業日の始値と終値をprice_open_1_day, price_close_1_day, ..., price_open_180_day, price_close_180_dayとして追加
        for i, (open_price, close_price) in enumerate(zip(daily_open_prices, daily_close_prices)):
            result[f"price_open_{i + 1}_day"] = open_price
            result[f"price_close_{i + 1}_day"] = close_price

        # 現在価格（最新の終値）
        result["current_price"] = history["Close"].iloc[-1] if not history.empty else None

        # 結果をリストに追加
        results.append(result)

    except Exception as e:
        print(f"Failed to process {ticker}: {e}")
        traceback.print_exc()  # エラーのスタックトレースを表示

# 結果をデータフレームに変換
results_df = pd.DataFrame(results)

# 結果を表示
print(results_df)

# 結果をCSVに保存
results_df.to_csv("ipo_open_close_180_unadjusted.csv", index=False)