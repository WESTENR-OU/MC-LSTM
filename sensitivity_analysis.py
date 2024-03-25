import pdb
import sys
import os
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

sys.path.append('../')
module_path = os.path.join(Path(__file__).parent, 'mc-lstm-main\mc-lstm-main')
sys.path.append(module_path)
sys.path.append('../')
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import load_data

basin_dict = {
    #"Delaware": {"basin_id": "01439500", "basin_area_km2": 306},
    #"Mill": {"basin_id": "06888500", "basin_area_km2": 843},
    #"Sycamore": {"basin_id": "09510200", "basin_area_km2": 428},
    #"Wimberley": {"basin_id": "08104900", "basin_area_km2": 348},
    #"Bull": {"basin_id": "06224000", "basin_area_km2": 484},
    #"Sauk": {"basin_id": "12189500", "basin_area_km2": 1849},
    #"Stevens": {"basin_id": "02196000", "basin_area_km2": 1412},
    "Suwannee": {"basin_id": "02314500", "basin_area_km2": 2926}
}
font = {'family': 'Arial', 'size': 14}
matplotlib.rc('font', **font)
hidden_nodes_num = 128
n_epochs = 200
batch_size = 64
fold_list = ['f1', 'f2', 'f3']

dfs = []
basin_names = []
learning_rate = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007,
                 0.0008, 0.0009, 0.001, 0.0012, 0.0015, 0.002, 0.003]

for basin_name in basin_dict:
    statistics_folder_path_LSTM = \
        r"C:\Users\yihan\OneDrive\OU\To_Yihan\To_Yihan\LSTM_test\result_128nodes\statistics_LSTM_sens\%s"%basin_name
    statistics_folder_path_MCLSTM = \
        r"C:\Users\yihan\OneDrive\OU\To_Yihan\To_Yihan\LSTM_test\result_128nodes\statistics_MC_sens\%s"%basin_name
    dfs_LSTM = []
    dfs_MCLSTM = []
    for lr in learning_rate:
        statistics_file_path_LSTM = os.path.join(statistics_folder_path_LSTM, 'LSTM_' + basin_name + '_%.5flr.csv' % lr)
        stats_df = pd.read_csv(statistics_file_path_LSTM)
        stats_df['lr'] = lr
        dfs_LSTM.append(stats_df)

        statistics_file_path_MCLSTM = os.path.join(statistics_folder_path_MCLSTM, 'MCLSTM_' + basin_name + '_%.5flr.csv' % lr)
        stats_df = pd.read_csv(statistics_file_path_MCLSTM)
        stats_df['lr'] = lr
        dfs_MCLSTM.append(stats_df)

    stats_df_all_LSTM = pd.concat(dfs_LSTM, ignore_index=True)
    stats_df_all_MCLSTM = pd.concat(dfs_MCLSTM, ignore_index=True)

# Choose the metrics you want to plot
metrics_to_plot = ['CC', 'KGE', 'NSE', 'RMSE', 'cumRB (%)']

# Define the number of rows and columns for the grid
num_rows = 2
num_cols = 3

# Create a grid of subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))

# Flatten the 2D array of axes to a 1D array
axes = axes.flatten()
if basin_name == 'Delaware':
    basin_name = 'Bushkill'
# Loop through metrics and plot in each subplot
for i, metric in enumerate(metrics_to_plot):
    metric_data_LSTM = stats_df_all_LSTM[stats_df_all_LSTM['Metric'] == metric]
    metric_data_MCLSTM = stats_df_all_MCLSTM[stats_df_all_MCLSTM['Metric'] == metric]

    # Plot the data on the corresponding subplot

    #sns.lineplot(x='lr', y='Cal', data=metric_data_LSTM, ax=axes[i],marker='o',markersize=8,
    #             linestyle='-', color='firebrick', linewidth=2)
    sns.lineplot(x='lr', y='Val', data=metric_data_LSTM, ax=axes[i],marker='o',markersize=8,
                 linestyle='-', color='firebrick', linewidth=2)
    #sns.lineplot(x='lr', y='Cal', data=metric_data_MCLSTM, ax=axes[i],marker='^',markersize=10,
    #             linestyle='-', color='deepskyblue', linewidth=2)
    sns.lineplot(x='lr', y='Val', data=metric_data_MCLSTM, ax=axes[i],marker='^',markersize=10,
                 linestyle='-', color='deepskyblue', linewidth=2)
    axes[i].set_xscale('log')  # Set log scale for x-axis
    axes[i].set_xlabel('Learning Rate')
    if metric == 'RMSE':
        metric = 'RMSE (m$^3$/s)'
    axes[i].set_ylabel(metric)
    # Set x-axis labels in scientific notation
    #axes[i].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    '''
    if metric in ['CC', 'KGE', 'NSE']:
        axes[i].set_ylim(top=1)
    '''
    #axes[i].legend()
    axes[i].grid(True)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()

plt.savefig('figureset_128nodes/sensitivity_%s.png'%basin_name, dpi=600)
