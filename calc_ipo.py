import pandas as pd
import yfinance as yf
from datetime import datetime
import traceback
from tqdm import tqdm  # tqdmライブラリをインポート

# CSVファイルを読み込む
csv_file = "ipo_list_2024.csv"
ipo_data = pd.read_csv(csv_file)

# 公開価格を数値に変換（カンマを削除してfloat型に変換）
ipo_data["public_price"] = ipo_data["public_price"].replace(",", "", regex=True).astype(float)

# 結果を格納するリスト
results = []

# デバグ用にループ回数を制限
loop_limit = 100  # ループ回数制限
loop_count = 0

# 各銘柄のデータを取得
for index, row in tqdm(ipo_data.iterrows(), total=len(ipo_data), desc="Processing IPO data"):
    ticker = row["ticker"]  # 銘柄コード
    public_price = row["public_price"]
    listing_date = row["listing_date"].strip()  # 上場日
    listing_date = datetime.strptime(listing_date, "%Y/%m/%d")  # 日付形式に変換

    try:
        # 株価データを取得
        stock = yf.Ticker(ticker)
        history = stock.history(start=listing_date, end=datetime.now(), auto_adjust=True, interval="1d")

        if history.empty:
            print(f"No data available for {ticker}.")
            continue

        # 初値（上場日の始値）
        first_open_price = history.iloc[0]["Open"]  # 最初の行の始値

        # 上場10営業日後の終値
        if len(history.index) > 9:  # 10営業日目が存在するか確認
            day_10 = history.index[9]  # 10営業日目のインデックス
            price_10_days = history.loc[day_10]["Close"]
        else:
            price_10_days = None

        # 上場30営業日後の終値
        if len(history.index) > 29:  # 30営業日目が存在するか確認
            day_30 = history.index[29]  # 30営業日目のインデックス
            price_30_days = history.loc[day_30]["Close"]
        else:
            price_30_days = None

        # 上場60営業日後の終値
        if len(history.index) > 59:  # 60営業日目が存在するか確認
            day_60 = history.index[59]  # 60営業日目のインデックス
            price_60_days = history.loc[day_60]["Close"]
        else:
            price_60_days = None
        
        # 上場90営業日後の終値
        if len(history.index) > 89:  # 90営業日目が存在するか確認
            day_90 = history.index[89]
            price_90_days = history.loc[day_90]["Close"]
        else:
            price_90_days = None

        # 上場180営業日後の終値
        if len(history.index) > 179:  # 180営業日目が存在するか確認
            day_180 = history.index[179]  # 180営業日目のインデックス
            price_180_days = history.loc[day_180]["Close"]
        else:
            price_180_days = None

        # 現在価格（最新の終値）
        current_price = history["Close"].iloc[-1] if not history.empty else None

        # 結果をリストに追加
        results.append({
            "ticker": ticker,
            "company_name": row["company_name"],
            "public_price": public_price,
            "first_open_price": first_open_price,
            "price_10_days": price_10_days,
            "price_30_days": price_30_days,
            "price_60_days": price_60_days,
            "price_90_days": price_90_days,
            "price_180_days": price_180_days,
            "current_price": current_price
        })
        loop_count += 1
        if loop_count >= loop_limit:  # デバッグ用にループ回数を制限
            break
        # print(f"Processed {ticker}: {results[-1]}")  # デバッグ用出力
        # break  # デバッグ用に1銘柄のみ処理
    except Exception as e:
        print(f"Failed to process {ticker}: {e}")
        traceback.print_exc()  # エラーのスタックトレースを表示

# 結果をデータフレームに変換
results_df = pd.DataFrame(results)

# 結果を表示
print(results_df)

# 結果をCSVに保存
results_df.to_csv("ipo_detailed_analysis.csv", index=False)

