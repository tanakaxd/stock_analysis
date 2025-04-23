import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
from tqdm import tqdm

# 1. 日経225採用企業リストの読み込み
file_path = "nikkei225_companies.csv"  # 日経225企業リストのCSVファイル
companies = pd.read_csv(file_path)

# 結果を格納するリスト
stable_growth_companies = []

# 2. 各企業の株価データを取得して分析
for index, row in tqdm(companies.iterrows(), total=len(companies), desc="Processing companies"):
    ticker = f"{row['ticker']}.T"  # 日本株式のティッカー形式（例: 7203.T）
    company_name = row['company_name']

    try:
        # 株価データを取得（過去10年以上）
        start_date = (datetime.now().year - 10, datetime.now().month, datetime.now().day)
        stock = yf.Ticker(ticker)
        history = stock.history(start=f"{start_date[0]}-{start_date[1]:02d}-{start_date[2]:02d}", end=datetime.now(), interval="1d")
        # print(f"Processing {company_name} length: ({len(history)})...")
        # print(history.head())  # データの最初の数行を表示
        # break
        

        if history.empty or len(history) < 2463:  # 2463営業日 ≈ 10年
            print(f"Insufficient data for {company_name} ({ticker}). Skipping.")
            continue

        # 3. 株価の変動率を計算
        history['daily_return'] = history['Close'].pct_change()  # 日次リターン
        volatility = history['daily_return'].std()  # 標準偏差（ボラティリティ）

        # 4. 長期トレンドを確認
        overall_return = (history['Close'].iloc[-1] - history['Close'].iloc[0]) / history['Close'].iloc[0]  # 全体のリターン
        annualized_return = (1 + overall_return) ** (1 / 10) - 1  # 年率リターン

        # 条件: ボラティリティが低く、年率リターンが正
        if volatility < 0.02 and annualized_return > 0.05:  # ボラティリティ < 2%, 年率リターン > 5%
            stable_growth_companies.append({
                "ticker": ticker,
                "company_name": company_name,
                "volatility": volatility,
                "annualized_return": annualized_return
            })

    except Exception as e:
        print(f"Failed to process {company_name} ({ticker}): {e}")

# 5. 結果をデータフレームに変換
results_df = pd.DataFrame(stable_growth_companies)

# 6. 結果をCSVに保存
output_file = "stable_growth_companies.csv"
results_df.to_csv(output_file, index=False)
print(f"安定成長企業のリストをCSVファイルに保存しました: {output_file}")

# 7. 結果を表示
print("安定成長企業のリスト:")
print(results_df)