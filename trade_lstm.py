import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# データ取得
# ticker = "AAPL"
# ticker = "6857.T" # アドバンテスト
# ticker = "7203.T" # トヨタ自動車
# ticker = "9984.T" # ソフトバンクグループ
# ticker = "8306.T" # 
# ticker = "8035.T"
# ticker = "7974.T"
# ticker = "9983.T"
# ticker = "6861.T"
# ticker = "6857.T"
ticker = "9432.T" # NTT

try:
    # data = yf.download(ticker, start="2020-01-01", end="2023-01-01", timeout=10)
    data = yf.download(ticker, interval="5m", start="2025-02-27", end="2025-04-27")  # 過去二か月の5分足データ
    if data.empty:
        raise ValueError("データが取得できませんでした。")
    # MultiIndexの場合、フラット化
    if isinstance(data.columns, pd.MultiIndex):
        print("MultiIndex detected, flattening...")
        data.columns = data.columns.get_level_values(0)
except Exception as e:
    print(f"データ取得エラー: {e}")
    exit(1)

# データが取得できたか確認
print(f"取得データ数: {len(data)}")
if len(data) < 100:  # 最低限のデータ数チェック
    print("データが不足しています。別の期間またはティッカーを試してください。")
    exit(1)

# データ準備
data['Returns'] = data['Close'].pct_change()
data['Future_Return'] = (data['Close'].shift(-5) - data['Close']) / data['Close']

# スケーリング
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close', 'Returns']].fillna(0))
sequence_length = 50

X, y = [], []
for i in range(len(scaled_data) - sequence_length - 5):
    X.append(scaled_data[i:i + sequence_length])
    y.append(data['Future_Return'].iloc[i + sequence_length])
X, y = np.array(X), np.array(y)

# 以下、元のスクリプトと同じ
# 訓練・テストデータ分割
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTMモデル構築
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 2)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 予測
predictions = model.predict(X_test)
data['Predicted_Return'] = pd.Series(np.concatenate([np.zeros(train_size + sequence_length + 5), predictions.flatten()]), index=data.index)

# シグナル生成
data['Signal'] = np.where(data['Predicted_Return'] > 0, 1, -1)
data['Strategy_Return'] = data['Signal'] * data['Future_Return']

# 取引コスト
transaction_cost = 0.001
data['Strategy_Return'] = data['Strategy_Return'] - transaction_cost * data['Signal'].diff().abs()

# 結果評価
cumulative_return = (1 + data['Strategy_Return'].fillna(0)).cumprod()
sharpe_ratio = data['Strategy_Return'].mean() / data['Strategy_Return'].std() * np.sqrt(252)

print(f"累積リターン: {cumulative_return.iloc[-1]:.2f}")
print(f"シャープレシオ: {sharpe_ratio:.2f}")

# プロット
plt.plot(cumulative_return)
plt.title(f"{ticker} LSTM Strategy")
plt.show()