import os
import pandas as pd
import yfinance as yf
from datetime import timedelta

# 1. 株価データの取得
ticker = "6857.T"  # アドバンテスト
# ticker = "7203.T"  # トヨタ自動車
# ticker = "6758.T"  # ソニーグループ
data = yf.download(ticker, interval="5m", start="2025-02-27", end="2025-04-25")  # 過去5日間の5分足データ
# data = yf.download(ticker, interval="5m", start="2025-04-22", end="2025-04-23")  # 過去5日間の5分足データ
if isinstance(data.columns, pd.MultiIndex):
    print("MultiIndex detected, flattening...")
    data.columns = data.columns.get_level_values(0)
# データの全行を出力
print("=== 取得したデータ ===")
print(data)


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
afternoon_end = pd.to_datetime("6:30").time()

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

# トレード戦略の実装（利幅と損切り幅をパラメータ化）
# 簡易的修正として損切りラインへの接触が常に利確ラインとの接触より先に起こると仮定してみる
def trade_strategy(data, target_profit, stop_loss):
    results = {
        "success": 0,
        "fail": 0,
        "total_profit": 0,
    }
    trades = []  # 各取引の詳細を記録するリスト

    # データを時系列順に処理
    data = data.reset_index()

    for i in range(1, len(data)):
        prev_close = data.loc[i - 1, "Close"]
        prev_open = data.loc[i - 1, "Open"]
        prev_candle = prev_close - prev_open  # 陽線か陰線かを判定

        # 現在のローソク足のデータ
        current_time = data.loc[i, "Datetime"] + timedelta(hours=9)  # 日本時間に変換
        current_open = data.loc[i, "Open"]
        current_high = data.loc[i, "High"]
        current_low = data.loc[i, "Low"]
        current_close = data.loc[i, "Close"]

        # 損益を計算
        profit_or_loss = 0
        trade_type = None  # 売り or 買い
        entry_price = current_open
        exit_price = None

        # 陽線の場合
        if prev_candle > 0:
            trade_type = "Sell"
            target_price = current_open - target_profit  # 利確幅
            stop_price = current_open + stop_loss  # 損切り幅
            if current_high >= stop_price:  # 損切りに到達
                exit_price = stop_price
                profit_or_loss = current_open - stop_price
                results["fail"] += 1
            elif current_low <= target_price:  # 利確に到達
                exit_price = target_price
                profit_or_loss = current_open - target_price
                results["success"] += 1
            else:  # 終値で決済
                exit_price = current_close
                profit_or_loss = current_open - current_close
                results["fail"] += 1

        # 陰線の場合
        elif prev_candle < 0:
            trade_type = "Buy"
            target_price = current_open + target_profit  # 利確幅
            stop_price = current_open - stop_loss  # 損切り幅
            if current_low <= stop_price:  # 損切りに到達
                exit_price = stop_price
                profit_or_loss = stop_price - current_open
                results["fail"] += 1
            elif current_high >= target_price:  # 利確に到達
                exit_price = target_price
                profit_or_loss = target_price - current_open
                results["success"] += 1
            else:  # 終値で決済
                exit_price = current_close
                profit_or_loss = current_close - current_open
                results["fail"] += 1

        # 合計損益を更新
        results["total_profit"] += profit_or_loss

        # Session情報を取得
        session = data.loc[i, "Session"]

        # 取引の詳細を記録
        trades.append({
            "Time (JST)": current_time,
            "Trade Type": trade_type,
            "Entry Price": entry_price,
            "Exit Price": exit_price,
            "Profit/Loss": profit_or_loss,
            "Target Reached": exit_price == target_price,
            "Stop Loss Triggered": exit_price == stop_price,
            "Session": session  # Sessionを追加
        })

    return results, trades

# 利確幅と損切り幅を変えて検証
target_profits = [i for i in range(0, 1)]  # 利幅のリスト
stop_losses = [i for i in range(0, 1)]  # 損切り幅

# 組み合わせごとの結果を記録するリスト
profit_results = []

for target_profit in target_profits:
    for stop_loss in stop_losses:
        # 利確幅と損切り幅を変えて検証
        print(f"\n=== 利確幅: {target_profit}円, 損切り幅: {stop_loss}円 ===")
        results, trades = trade_strategy(data, target_profit, stop_loss)

        # 結果を表示
        print("Results:")
        print(f"Success: {results['success']}, Fail: {results['fail']}")
        print(f"Total Profit: {results['total_profit']}")

        # 組み合わせごとの結果を記録
        profit_results.append({
            "Target Profit": target_profit,
            "Stop Loss": stop_loss,
            "Total Profit": results["total_profit"],
            "Success Trades": results["success"],
            "Failed Trades": results["fail"]
        })

        # 必要に応じて取引データをDataFrameに変換
        trades_df = pd.DataFrame(trades)

        # 日付と時間を分離
        trades_df["Date"] = trades_df["Time (JST)"].dt.date
        trades_df["Time"] = trades_df["Time (JST)"].dt.time
        trades_df = trades_df.drop(columns=["Time (JST)"])

        # ファイル名を生成
        ticker_num = ticker.split(".")[0]
        output_dir = f"candle_switch_trade_data_{ticker_num}_losscut_first"

        # ディレクトリが存在しない場合は作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"trade_data_{ticker_num}_{target_profit}yen_{stop_loss}yen.csv")

        # データをCSVに保存
        trades_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"Trade data saved to {output_file}")

# 組み合わせごとの結果をDataFrameに変換
profit_results_df = pd.DataFrame(profit_results)

# 結果をCSVに保存
# output_file = os.path.join(output_dir, f"profit_results_{ticker_num}.csv")
# profit_results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
# print(f"Profit results saved to {output_file}")