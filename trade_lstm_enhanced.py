import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas_ta as ta

# データ取得
ticker = "9432.T"  
try:
    data = yf.download(ticker, interval="5m", start="2025-02-27", end="2025-04-27")
    if data.empty:
        raise ValueError("データが取得できませんでした。")
    if isinstance(data.columns, pd.MultiIndex):
        print("MultiIndex detected, flattening...")
        data.columns = data.columns.get_level_values(0)
except Exception as e:
    print(f"データ取得エラー: {e}")
    exit(1)

# データが取得できたか確認
print(f"取得データ数: {len(data)}")
if len(data) < 100:
    print("データが不足しています。別の期間またはティッカーを試してください。")
    exit(1)

# テクニカル指標の計算
data['Returns'] = data['Close'].pct_change()
data['Future_Return'] = (data['Close'].shift(-5) - data['Close']) / data['Close']
data['RSI'] = ta.rsi(data['Close'], length=14)
data['MACD'] = ta.macd(data['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
data['BB_Upper'] = ta.bbands(data['Close'], length=20, std=2)['BBU_20_2.0']
data['BB_Lower'] = ta.bbands(data['Close'], length=20, std=2)['BBL_20_2.0']

# センチメントプロキシ
data['Price_Spread'] = (data['High'] - data['Low']) / data['Close']
data['Sentiment_Proxy'] = data['Price_Spread'] * data['Volume']

# データ準備
features = ['Close', 'Returns', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Sentiment_Proxy']
data = data.dropna()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])
sequence_length = 50

X, y = [], []
for i in range(len(scaled_data) - sequence_length - 5):
    X.append(scaled_data[i:i + sequence_length])
    y.append(data['Future_Return'].iloc[i + sequence_length])
X, y = np.array(X), np.array(y)

# 訓練・テストデータ分割
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTMモデル構築
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# 予測
predictions = model.predict(X_test)
data['Predicted_Return'] = pd.Series(np.concatenate([np.zeros(train_size + sequence_length + 5), predictions.flatten()]), index=data.index)

# ポジションサイジング
max_position = 2.0
data['Position_Size'] = np.clip(np.abs(data['Predicted_Return']) * 100, 0, max_position) * np.sign(data['Predicted_Return'])
data['Position_Size'] = np.where(np.abs(data['Position_Size']) < 0.1, 0, data['Position_Size'])

# 戦略リターン計算
data['Strategy_Return'] = data['Position_Size'] * data['Future_Return']
transaction_cost = 0.0001
data['Strategy_Return'] = data['Strategy_Return'] - transaction_cost * data['Position_Size'].diff().abs()

# 結果評価
cumulative_return = (1 + data['Strategy_Return'].fillna(0)).cumprod()
sharpe_ratio = data['Strategy_Return'].mean() / data['Strategy_Return'].std() * np.sqrt(252 * 12 * 5.5) # 5.5時間の取引時間を考慮
print(f"累積リターン: {cumulative_return.iloc[-1]:.2f}")
print(f"シャープレシオ: {sharpe_ratio:.2f}")
print(f"取引回数: {int(data['Position_Size'].diff().abs().gt(0).sum())}")
print(f"Strategy Return Mean: {data['Strategy_Return'].mean():.6f}")
print(f"Strategy Return Std: {data['Strategy_Return'].std():.6f}")
print(data['Predicted_Return'].describe())

# プロット
plt.figure(figsize=(12, 6))
plt.plot(cumulative_return, label='Strategy Cumulative Return')
plt.title(f"{ticker} Enhanced LSTM Strategy")
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.savefig('strategy_performance.png')
plt.close()