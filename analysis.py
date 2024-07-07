import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import glob
import os
import re


os.makedirs('output/insight', exist_ok=True)


def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return list(map(int, numbers))


reward_file_paths = glob.glob('output/*/reward.csv')
reward_sorted_file_paths = sorted(reward_file_paths, key=numerical_sort)

reward_list = []

for path in reward_sorted_file_paths:
    df = pd.read_csv(path, index_col=0)
    reward_df = df.sum(axis=0)
    reward = float(reward_df.mean())
    reward_list.append(reward)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(reward_list)
ax.set_title('Reward')
ax.set_xlabel('Episode')
# ax.set_ylabel('Reward')
# plt.show()
fig.savefig('output/insight/reward.png', dpi=300)

# ==================================================================================================
cost_file_paths = glob.glob('output/*/electricity_cost.csv')
cost_sorted_file_paths = sorted(cost_file_paths, key=numerical_sort)

cost_list = []

for path in cost_sorted_file_paths:
    df = pd.read_csv(path, index_col=0)
    cost_df = df.sum(axis=0)
    cost_list.append(float(cost_df.mean())/100)

print(cost_list)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(cost_list)
ax.set_title('Cost')
ax.set_xlabel('Episode')
ax.set_ylabel('Cost [$]')
# plt.show()
fig.savefig('output/insight/cost.png', dpi=300)


# ==================================================================================================
grid_import_file_paths = glob.glob('output/*/grid_import_record.csv')
grid_import_sorted_file_paths = sorted(grid_import_file_paths, key=numerical_sort)

grid_import_sum_list = []

for path in grid_import_sorted_file_paths:
    df = pd.read_csv(path, index_col=0)
    grid_import_sum_list.append(df.sum(axis=0).values[0])

# データポイントの作成
x = np.arange(len(grid_import_sum_list))
y = np.array(grid_import_sum_list)

# 線形回帰の計算
coefficients = np.polyfit(x, y, 1)
polynomial = np.poly1d(coefficients)
trendline = polynomial(x)

# グラフの作成
fig, ax = plt.subplots(figsize=(8, 6))

# データポイントのプロット
ax.plot(x, y, label='Data Points')

# トレンドラインのプロット
ax.plot(x, trendline, label='Trend Line', color='blue')

# グラフの装飾
ax.set_xlabel('Episode')
ax.set_ylabel('Grid Import [kWh]')
ax.set_title('Grid Import')
ax.legend()

# グラフの表示
# plt.show()

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(grid_import_sum_list)
# ax.set_title('Grid Import')
# ax.set_xlabel('Episode')
# ax.set_ylabel('Grid Import [kWh]')
# plt.show()
fig.savefig('output/insight/grid_import.png', dpi=300)