import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt

# パラメータ
DATA_DIR = 'data_utf8'
MODEL_DIR = 'model'
COMBINED_DATA_PATH = os.path.join(MODEL_DIR, 'nikkei_combined_5min.csv')
SEQUENCE_LENGTH = 50
LOOKAHEAD_PERIOD = 10
SCALING_FACTOR = 200
THRESHOLD = 0.05
FEATURES = ['Close', 'Returns', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Sentiment_Proxy']
MODEL_PATH = os.path.join(MODEL_DIR, 'nikkei_lstm_model.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LOSS_PLOT_PATH = os.path.join(MODEL_DIR, 'loss_plot.png')

# モデルディレクトリを作成
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. データ取得・統合
def load_and_combine_data(data_dir, combined_data_path):
    def load_data(file_path):
        df = pd.read_csv(file_path, parse_dates=['datetime'])
        if df.columns[0].startswith('Unnamed') or df.columns[0] == '':
            df = df.drop(columns=df.columns[0])
        df = df.set_index('datetime')
        if df.index.duplicated().any():
            print(f"Found {df.index.duplicated().sum()} duplicate timestamps. Dropping duplicates.")
            df = df[~df.index.duplicated(keep='first')]
        df = df.dropna()
        return df
    
    if os.path.exists(combined_data_path):
        print(f"結合済みデータ {combined_data_path} を読み込み中...")
        combined_df = load_data(combined_data_path)
        return combined_df
    
    all_data = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', header=[0, 1])
                    df = df.iloc[:77, [0, 15, 16, 17, 18, 19, 20]]
                    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
                    all_data.append(df)
                except Exception as e:
                    print(f"エラー: {file_path} - {e}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    def parse_date(date_str):
        try:
            return pd.to_datetime(date_str, format='%y/%m/%d')
        except:
            try:
                return pd.to_datetime(date_str, format='%Y/%m/%d')
            except:
                print(f"日付パースエラー: {date_str}")
                return pd.NaT
    
    combined_df['date'] = combined_df['date'].apply(parse_date)
    combined_df = combined_df.dropna(subset=['date'])
    combined_df['datetime'] = pd.to_datetime(combined_df['date'].astype(str) + ' ' + combined_df['time'])
    
    # Reduce dataset span to 2024 to match trade_lstm_grid_search.py
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2024-11-30')
    combined_df = combined_df[(combined_df['datetime'] >= start_date) & (combined_df['datetime'] <= end_date)]
    combined_df = combined_df.sort_values('datetime')
    combined_df = combined_df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    
    print(f"結合済みデータを {combined_data_path} に保存中...")
    combined_df.to_csv(combined_data_path, index=True)
    
    return combined_df

# 2. 特徴量エンジニアリング
def engineer_features(df):
    df = df.copy()
    
    df['Close'] = df['close']
    df['Returns'] = df['close'].pct_change()
    df['Future_Return'] = (df['close'].shift(-LOOKAHEAD_PERIOD) - df['close']) / df['close']
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['MACD'] = ta.macd(df['close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_Upper'] = bb['BBU_20_2.0']
    df['BB_Lower'] = bb['BBL_20_2.0']
    df['Price_Spread'] = (df['high'] - df['low']) / df['close']
    df['Sentiment_Proxy'] = df['Price_Spread'] * df['volume']
    
    # Handle NaN and inf in all features and Future_Return
    df[FEATURES + ['Future_Return']] = df[FEATURES + ['Future_Return']].replace([np.inf, -np.inf], np.nan)
    df[FEATURES + ['Future_Return']] = df[FEATURES + ['Future_Return']].interpolate(method='linear').fillna(method='bfill')
    
    # Drop any remaining NaN
    df = df.dropna(subset=FEATURES + ['Future_Return'])
    
    return df

# 3. シーケンスデータ作成
def create_sequences(data, seq_length, lookahead):
    X, y = [], []
    for i in range(len(data) - seq_length - lookahead + 1):
        X.append(data[i:(i + seq_length), :len(FEATURES)])  # Exclude Future_Return from X
        y.append(data[i + seq_length + lookahead - 1, len(FEATURES)])  # Future_Return as y
    X, y = np.array(X), np.array(y)
    
    # Check for NaN or inf in X and y
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError(f"NaN or inf found in X: {np.isnan(X).sum()} NaN, {np.isinf(X).sum()} inf")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError(f"NaN or inf found in y: {np.isnan(y).sum()} NaN, {np.isinf(y).sum()} inf")
    
    return X, y

# メイン処理
def main():
    print("データを読み込み中...")
    df = load_and_combine_data(DATA_DIR, COMBINED_DATA_PATH)
    
    print("特徴量を計算中...")
    df = engineer_features(df)
    
    # Debug: Check for NaN and inf
    print(f"NaN in DataFrame:\n{df[FEATURES + ['Future_Return']].isna().sum()}")
    print(f"Inf in Future_Return: {np.isinf(df['Future_Return']).sum()}")
    print(f"Future_Return distribution:\n{df['Future_Return'].describe()}")
    
    # Separate scaling for features and target
    feature_data = df[FEATURES].values
    target_data = df[['Future_Return']].values
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler(feature_range=(-1, 1))  # Increase variance
    feature_data = feature_scaler.fit_transform(feature_data)
    target_data = target_scaler.fit_transform(target_data)
    feature_data = np.hstack([feature_data, target_data])
    
    # Save both scalers
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({'feature': feature_scaler, 'target': target_scaler}, f)
    
    print("シーケンスデータを作成中...")
    X, y = create_sequences(feature_data, SEQUENCE_LENGTH, LOOKAHEAD_PERIOD)
    
    # Debug: Check shapes and target distribution
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"y distribution:\n{pd.Series(y).describe()}")
    
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Debug: Check for NaN in train/val sets
    print(f"NaN in y_train: {np.isnan(y_train).sum()}, NaN in y_val: {np.isnan(y_val).sum()}")
    
    print("モデルを構築中...")
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(FEATURES))),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    # 学習率を下げて、より細かい学習を可能に
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    
    print("モデルを学習中...")
    # 早期停止の条件を緩和（patienceを増やし、min_deltaを追加）
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.0001,
        restore_best_weights=True
    )
    
    # バッチサイズを小さくして、より細かい学習を可能に
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # エポック数を増やす
        batch_size=16,  # バッチサイズを小さく
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()
    
    print("SHAP分析をスキップしました。")
    
    print(f"モデルを保存中: {MODEL_PATH}")
    model.save(MODEL_PATH)
    
    print("処理完了！")
    print(f"結合済みデータ: {COMBINED_DATA_PATH}")
    print(f"モデル: {MODEL_PATH}")
    print(f"スケーラー: {SCALER_PATH}")
    print(f"損失プロット: {LOSS_PLOT_PATH}")

if __name__ == '__main__':
    main()