import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('optimization_results.csv')

# Data preprocessing
df = df.dropna()  # Remove rows with missing values
df = df[df['cumulative_return'] > 0]  # Remove invalid returns
df['ticker'] = df['ticker'].astype(str)
df['sequence_length'] = df['sequence_length'].astype(int)
df['lookahead_period'] = df['lookahead_period'].astype(int)
df['scaling_factor'] = df['scaling_factor'].astype(int)
df['threshold'] = df['threshold'].astype(float)

# Correlation analysis
numeric_cols = ['sequence_length', 'lookahead_period', 'scaling_factor', 'threshold', 
                'cumulative_return', 'sharpe_ratio', 'trade_count', 'total_cost', 'pred_return_std']
correlation_matrix = df[numeric_cols].corr()

# Save correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Parameters and Metrics')
plt.savefig('correlation_heatmap.png')
plt.close()

# Grouped analysis
grouped_stats = {}
for param in ['ticker', 'sequence_length', 'lookahead_period', 'scaling_factor', 'threshold']:
    grouped_stats[param] = df.groupby(param)['cumulative_return'].agg(['mean', 'median', 'std']).reset_index()

# Box plots for cumulative return by parameter
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

# Ticker
sns.boxplot(x='ticker', y='cumulative_return', data=df, ax=axes[0])
axes[0].set_title('Cumulative Return by Ticker')
axes[0].tick_params(axis='x', rotation=45)

# Sequence Length
sns.boxplot(x='sequence_length', y='cumulative_return', data=df, ax=axes[1])
axes[1].set_title('Cumulative Return by Sequence Length')

# Lookahead Period
sns.boxplot(x='lookahead_period', y='cumulative_return', data=df, ax=axes[2])
axes[2].set_title('Cumulative Return by Lookahead Period')

# Scaling Factor
sns.boxplot(x='scaling_factor', y='cumulative_return', data=df, ax=axes[3])
axes[3].set_title('Cumulative Return by Scaling Factor')

# Threshold
sns.boxplot(x='threshold', y='cumulative_return', data=df, ax=axes[4])
axes[4].set_title('Cumulative Return by Threshold')

# Remove empty subplot
fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig('box_plots.png')
plt.close()

# Heatmap for sequence_length vs lookahead_period
pivot_table = df.pivot_table(values='cumulative_return', index='sequence_length', columns='lookahead_period', aggfunc='mean')
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.3f')
plt.title('Mean Cumulative Return: Sequence Length vs Lookahead Period')
plt.savefig('heatmap_seq_lookahead.png')
plt.close()

# Identify top parameter sets
top_params = df[
    (df['cumulative_return'] > 1.1) & 
    (df['sharpe_ratio'] > 2) & 
    # (df['trade_count'] < 300) & 
    (df['total_cost'] < 0.02)
].sort_values('cumulative_return', ascending=False).head(20)

# Save top parameter sets to CSV
top_params.to_csv('top_parameter_sets.csv', index=False)

# Print analysis results
print("=== Parameter Analysis ===")
print("\nCorrelation with Cumulative Return:")
print(correlation_matrix['cumulative_return'].sort_values(ascending=False))

for param in grouped_stats:
    print(f"\nCumulative Return by {param}:")
    print(grouped_stats[param].to_string(index=False))

print("\n=== Top 5 Parameter Sets ===")
print(top_params[['ticker', 'sequence_length', 'lookahead_period', 'scaling_factor', 'threshold', 
                 'cumulative_return', 'sharpe_ratio', 'trade_count', 'total_cost']].to_string(index=False))

# Save grouped stats to CSV for reference
for param, stats in grouped_stats.items():
    stats.to_csv(f'grouped_stats_{param}.csv', index=False)

print("\nPlots saved: correlation_heatmap.png, box_plots.png, heatmap_seq_lookahead.png")
print("Top parameter sets saved to top_parameter_sets.csv")