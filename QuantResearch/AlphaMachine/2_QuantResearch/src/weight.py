import pandas as pd
import numpy as np

closes = np.array([5, 15, 66, 85, 25])
stocks = ['Stock A', 'Stock B', 'Stock C', 'Stock D', 'Stock E']

raw_alpha = -closes

print(f"原始 Alpha (-close): {raw_alpha}")

mean_val = np.mean(raw_alpha)
neutralized_alpha = raw_alpha - mean_val

print(f"Alpha 均值: {mean_val}")
print(f"中性化后的 Alpha: {neutralized_alpha}")

abs_sum = np.sum(np.abs(neutralized_alpha))
weights = neutralized_alpha / abs_sum

long_count = np.sum(weights > 0)
short_count = np.sum(weights < 0)

df_result = pd.DataFrame({
    'Close': closes,
    'Raw_Alpha': raw_alpha,
    'Neutralized_Alpha': neutralized_alpha,
    'Final_Weight': weights,
    'Direction': ['Long' if w > 0 else 'Short' for w in weights]
})

print("-" * 30)
print("计算详情表：")
print(df_result)
print("-" * 30)
print(f"Long Count (做多数量): {long_count}")
print(f"Short Count (做空数量): {short_count}")
