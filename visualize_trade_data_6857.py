import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the CSV file
df = pd.read_csv('candle_switch_trade_data_6857_losscut_first\profit_results_6857.csv', encoding='utf-8-sig')

# Extract unique values for Target Profit and Stop Loss
target_profits = np.sort(df['Target Profit'].unique())
stop_losses = np.sort(df['Stop Loss'].unique())

# Create a grid for Total Profit
total_profit_grid = np.zeros((len(target_profits), len(stop_losses)))
for i, tp in enumerate(target_profits):
    for j, sl in enumerate(stop_losses):
        total_profit_grid[i, j] = df[(df['Target Profit'] == tp) & (df['Stop Loss'] == sl)]['Total Profit'].values[0]

# Create a filled contour plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(stop_losses, target_profits, total_profit_grid, cmap='viridis', levels=20)
plt.colorbar(contour, label='Total Profit')
plt.xlabel('Stop Loss')
plt.ylabel('Target Profit')
plt.title('Total Profit vs Target Profit and Stop Loss')
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot
plt.savefig('profit_contour.png', dpi=300, bbox_inches='tight')
plt.close()