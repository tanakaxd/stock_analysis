import pandas as pd
import yfinance as yf
from datetime import timedelta

# 1. 株価データの取得
ticker = "6857.T"  # アドバンテスト
# ticker = "7203.T"  # トヨタ自動車
# ticker = "6758.T"  # ソニーグループ
data = yf.download(ticker, interval="5m", start="2025-02-23", end="2025-04-23")  # 過去5日間の5分足データ
if isinstance(data.columns, pd.MultiIndex):
    print("MultiIndex detected, flattening...")
    data.columns = data.columns.get_level_values(0)

# データが空の場合は終了
if data.empty:
    print(f"No data available for {ticker}.")
    exit()

# 2. 前場と後場の時間を定義
data["Time"] = data.index.time  # 時間を抽出
data["Date"] = data.index.date  # 日付を抽出
morning_start = pd.to_datetime("0:00").time()
morning_end = pd.to_datetime("2:30").time()
afternoon_start = pd.to_datetime("3:30").time()
afternoon_end = pd.to_datetime("6:00").time()

# 前場と後場に分割
def assign_session(time):
    if morning_start <= time <= morning_end:
        return "Morning"
    elif afternoon_start <= time <= afternoon_end:
        return "Afternoon"
    else:
        return None

# Session列を作成
data["Session"] = data["Time"].apply(assign_session)

# 前場・後場以外のデータを削除
data = data.dropna(subset=["Session"])

# 3. トレード戦略の実装（利幅をパラメータ化）
def trade_strategy(data, target_profit):
    results = {
        "Morning": {"success": 0, "fail": 0, "total_profit": 0},
        "Afternoon": {"success": 0, "fail": 0, "total_profit": 0},
    }
    trades = []  # 各取引の詳細を記録するリスト

    for session in ["Morning", "Afternoon"]:
        session_data = data[data["Session"] == session]
        session_data = session_data.reset_index()

        for i in range(1, len(session_data)):
            prev_close = session_data.loc[i - 1, "Close"]
            prev_open = session_data.loc[i - 1, "Open"]
            prev_candle = prev_close - prev_open  # 陽線か陰線かを判定

            # 現在のローソク足のデータ
            current_time = session_data.loc[i, "Datetime"] + timedelta(hours=9)  # 日本時間に変換
            current_open = session_data.loc[i, "Open"]
            current_high = session_data.loc[i, "High"]
            current_low = session_data.loc[i, "Low"]
            current_close = session_data.loc[i, "Close"]

            # 損益を計算
            profit_or_loss = 0
            trade_type = None  # 売り or 買い
            entry_price = current_open
            exit_price = None

            # 陽線の場合
            if prev_candle > 0:
                trade_type = "Sell"
                target_price = current_open - target_profit  # 利幅を動的に設定
                if current_low <= target_price:  # 利幅に到達
                    exit_price = target_price
                    profit_or_loss = current_open - target_price
                    results[session]["success"] += 1
                else:  # 終値で決済
                    exit_price = current_close
                    profit_or_loss = current_open - current_close
                    results[session]["fail"] += 1

            # 陰線の場合
            elif prev_candle < 0:
                trade_type = "Buy"
                target_price = current_open + target_profit  # 利幅を動的に設定
                if current_high >= target_price:  # 利幅に到達
                    exit_price = target_price
                    profit_or_loss = target_price - current_open
                    results[session]["success"] += 1
                else:  # 終値で決済
                    exit_price = current_close
                    profit_or_loss = current_close - current_open
                    results[session]["fail"] += 1

            # 合計損益を更新
            results[session]["total_profit"] += profit_or_loss

            # 取引の詳細を記録
            trades.append({
                "Session": session,
                "Time (JST)": current_time,
                "Trade Type": trade_type,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Profit/Loss": profit_or_loss,
                "Target Reached": exit_price == target_price
            })

    return results, trades

# 利幅を変えて検証
target_profits = [i for i in range(15,31)]  # 利幅のリスト
# 2n(n=1,2...5) の配列を生成
# target_profits = [n for n in range(1, 11)]  # 1から10までの整数を生成

for target_profit in target_profits:
    print(f"\n=== 利幅: {target_profit}円 ===")
    results, trades = trade_strategy(data, target_profit)

    # 結果を表示
    print("Morning Session Results:")
    print(f"Success: {results['Morning']['success']}, Fail: {results['Morning']['fail']}")
    print(f"Total Profit: {results['Morning']['total_profit']}")

    print("Afternoon Session Results:")
    print(f"Success: {results['Afternoon']['success']}, Fail: {results['Afternoon']['fail']}")
    print(f"Total Profit: {results['Afternoon']['total_profit']}")

    # 必要に応じて取引データをCSVに保存
    trades_df = pd.DataFrame(trades)
    trades_df["Date"] = trades_df["Time (JST)"].dt.date
    trades_df["Time"] = trades_df["Time (JST)"].dt.time
    trades_df = trades_df.drop(columns=["Time (JST)"])
    ticker_num = ticker.split(".")[0]
    output_file = f"trade_data_{ticker_num}_{target_profit}yen.csv"
    trades_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Trade data saved to {output_file}")