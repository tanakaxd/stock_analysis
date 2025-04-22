from scipy.stats import norm

# 観測された確率と仮定された確率
observed_probability = 0.4705  # 47.05%
expected_probability = 0.4    # 40%
sample_size = 13470           # サンプルサイズ

# z統計量を計算
z = (observed_probability - expected_probability) / ((expected_probability * (1 - expected_probability)) / sample_size) ** 0.5

# 両側検定のp値を計算
p_value = 2 * (1 - norm.cdf(abs(z)))

# 結果を出力
print(f"z統計量: {z:.4f}")
print(f"p値: {p_value:.4f}")

# 有意性の判定
alpha = 0.05  # 有意水準
if p_value < alpha:
    print("観測された確率は仮定された確率と有意に異なります（帰無仮説を棄却）。")
else:
    print("観測された確率は仮定された確率と有意に異ならない（帰無仮説を採択）。")