import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import shap
import matplotlib.pyplot as plt

# パラメータ
DATA_DIR = 'data_utf8'
MODEL_DIR = 'model'
COMBINED_DATA_PATH = os.path.join(MODEL_DIR, 'nikkei_combined_5min.csv')
SEQUENCE_LENGTH = 100
LOOKAHEAD_PERIOD = 10
SCALING_FACTOR = 200
THRESHOLD = 0.2
FEATURES = ['Close', 'Returns', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Sentiment_Proxy']
MODEL_PATH = os.path.join(MODEL_DIR, 'nikkei_lstm_model.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
SHAP_PLOT_PATH = os.path.join(MODEL_DIR, 'shap_summary.png')

# モデルディレクトリを作成
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. データ取得・統合
def load_and_combine_data(data_dir, combined_data_path):
    # 結合済みCSVが存在する場合、直接読み込む
    if os.path.exists(combined_data_path):
        print(f"結合済みデータ {combined_data_path} を読み込み中...")
        combined_df = pd.read_csv(combined_data_path, parse_dates=['datetime'], index_col='datetime')
        return combined_df
    
    # 全ファイルを探索して結合
    all_data = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    # CSV読み込み（ヘッダー2行、UTF-8）
                    df = pd.read_csv(file_path, encoding='utf-8', header=[0, 1])
                    # 必要な列を抽出（col1 + P:U列）
                    df = df.iloc[:77, [0, 15, 16, 17, 18, 19, 20]]  # 0–76行目
                    # マルチインデックスをフラット化
                    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
                    all_data.append(df)
                except Exception as e:
                    print(f"エラー: {file_path} - {e}")
    
    # データフレームを結合
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 日付をパース（20/11/19 または 2020/11/19）
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
    
    # 欠損日付を除外
    combined_df = combined_df.dropna(subset=['date'])
    
    # 日付と時刻を結合してdatetime列を作成
    combined_df['datetime'] = pd.to_datetime(combined_df['date'].astype(str) + ' ' + combined_df['time'])
    
    # 学習範囲を限定（2020/1/1～2024/11/30）
    start_date = pd.to_datetime('2020-01-01')
    end_date = pd.to_datetime('2024-11-30')
    combined_df = combined_df[(combined_df['datetime'] >= start_date) & (combined_df['datetime'] <= end_date)]
    
    # 時系列順にソート
    combined_df = combined_df.sort_values('datetime')
    
    # 不要な列を削除
    combined_df = combined_df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    
    # 結合済みデータをCSVに保存
    print(f"結合済みデータを {combined_data_path} に保存中...")
    combined_df.to_csv(combined_data_path, index=True)
    
    return combined_df

# 2. 特徴量エンジニアリング
def engineer_features(df):
    df = df.copy()
    
    # 特徴量計算
    df['Close'] = df['close']
    df['Returns'] = df['close'].pct_change()
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['MACD'] = ta.macd(df['close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_Upper'] = bb['BBU_20_2.0']
    df['BB_Lower'] = bb['BBL_20_2.0']
    
    # Sentiment_Proxy
    df['Price_Spread'] = (df['high'] - df['low']) / df['close']
    df['Sentiment_Proxy'] = df['Price_Spread'] * df['volume']
    
    # 欠損値処理（線形補間）
    df[FEATURES] = df[FEATURES].interpolate(method='linear').fillna(method='bfill')
    
    return df

# 3. シーケンスデータ作成
def create_sequences(data, seq_length, lookahead):
    X, y = [], []
    for i in range(len(data) - seq_length - lookahead + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length + lookahead - 1, 1])  # Returnsをターゲット
    return np.array(X), np.array(y)

# 4. SHAP分析
def analyze_feature_importance(model, X_val, feature_names):
    # SHAP Explainer（DeepExplainerを使用）
    explainer = shap.DeepExplainer(model, X_val[:100])  # 最初の100サンプルで計算を高速化
    shap_values = explainer.shap_values(X_val[:100])
    
    # SHAPサマリープロットを保存
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[0], X_val[:100], feature_names=feature_names, show=False)
    plt.savefig(SHAP_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 特徴量の重要度を出力
    shap_importance = np.abs(shap_values[0]).mean(axis=(0, 1))
    for i, (name, importance) in enumerate(zip(feature_names, shap_importance)):
        print(f"{name}: {importance:.4f}")

# メイン処理
def main():
    # データ読み込み
    print("データを読み込み中...")
    df = load_and_combine_data(DATA_DIR, COMBINED_DATA_PATH)
    
    # 特徴量エンジニアリング
    print("特徴量を計算中...")
    df = engineer_features(df)
    
    # 特徴量を選択
    feature_data = df[FEATURES].values
    
    # 正規化
    scaler = StandardScaler()
    feature_data = scaler.fit_transform(feature_data)
    
    # スケーラーを保存
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    # シーケンスデータ作成
    print("シーケンスデータを作成中...")
    X, y = create_sequences(feature_data, SEQUENCE_LENGTH, LOOKAHEAD_PERIOD)
    
    # 訓練/検証データに分割（80:20）
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # LSTMモデル構築
    print("モデルを構築中...")
    model = Sequential([
        LSTM(128, input_shape=(SEQUENCE_LENGTH, len(FEATURES)), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Returnsを予測
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # 学習
    print("モデルを学習中...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # SHAP分析
    print("特徴量の重要度を分析中...")
    analyze_feature_importance(model, X_val, FEATURES)
    
    # モデル保存
    print(f"モデルを保存中: {MODEL_PATH}")
    model.save(MODEL_PATH)
    
    print("処理完了！")
    print(f"結合済みデータ: {COMBINED_DATA_PATH}")
    print(f"モデル: {MODEL_PATH}")
    print(f"スケーラー: {SCALER_PATH}")
    print(f"SHAPプロット: {SHAP_PLOT_PATH}")

if __name__ == '__main__':
    main()