import pdb
import sys
import os
from pathlib import Path

import np as np

sys.path.append('../')
module_path = os.path.join(Path(__file__).parent, 'mc-lstm-main\mc-lstm-main')
sys.path.append(module_path)
sys.path.append('../')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import load_data

basin_dict = {
    "Delaware": {"basin_id": "01439500", "basin_area_km2": 306},
    "Mill": {"basin_id": "06888500", "basin_area_km2": 843},
    "Sycamore": {"basin_id": "09510200", "basin_area_km2": 428},
    "Wimberley": {"basin_id": "08104900", "basin_area_km2": 348},
    "Bull": {"basin_id": "06224000", "basin_area_km2": 484},
    "Sauk": {"basin_id": "12189500", "basin_area_km2": 1849},
    "Stevens": {"basin_id": "02196000", "basin_area_km2": 1412},
    "Suwannee": {"basin_id": "02314500", "basin_area_km2": 2926}
}
hidden_nodes_num = 128
n_epochs = 200
learning_rate = 0.0005
batch_size = 64
fold_list = ['f1', 'f2', 'f3']

dfs = []
basin_names = []
for basin_name in basin_dict:
    if basin_name in ['Mill', 'Wimberley',
                      'Sauk', 'Stevens']:
        learning_rate = 0.001
    elif basin_name == 'Delaware':
        learning_rate = 0.003
    elif basin_name == 'Sycamore':
        learning_rate = 0.0005
    elif basin_name == "Suwannee":
        learning_rate = 0.0006
    elif basin_name == 'Bull':
        learning_rate = 0.0001

    Data_folder = os.path.join(Path(__file__).parent.parent, 'Collaborative_Research\Data')

    ppt_path = Data_folder + "/" + basin_name + ".csv"
    PET_path = Data_folder + "/" + basin_name + "_PET.csv"
    # Obs path
    obs_path = Data_folder + "/" + basin_name + "_Obs.xlsx"

    df_ppt = load_data.load_forcing(ppt_path)
    df_PET = load_data.load_forcing(PET_path)
    df_obs = load_data.load_discharge(obs_path)

    #folder_path = os.path.join(os.path.dirname(__file__), 'result/timeseries_result/' + basin_name + '_hidden')
    folder_path = os.path.join(os.path.dirname(__file__), 'result_128nodes/timeseries_result_best/' + basin_name)

    basin_id = basin_dict[basin_name]["basin_id"]
    basin_area = basin_dict[basin_name]["basin_area_km2"]

    # ============================================#
    # compute the long-term mean ppt and PET     #
    # ============================================#
    df_ppt['date'] = pd.to_datetime(df_ppt['date'])
    df_ppt['year'] = df_ppt['date'].dt.year
    # Group by 'year' and 'month' and calculate the sum of the 'value' column for each month
    annual_sum_ppt = df_ppt.groupby(['year'])['value'].sum().reset_index()
    # Filter the years from 2002 to 2019
    selected_years = range(2002, 2020)
    filtered_sum = annual_sum_ppt[annual_sum_ppt['year'].isin(selected_years)]
    # Compute the mean of the monthly sums for the years 2002 to 2019
    mean_ppt = filtered_sum['value'].mean()
    # mean_ppt = mean_ppt * basin_area / 86.4
    # ============================================#
    # compute the long-term mean ppt and PET     #
    # ============================================#
    df_PET['date'] = pd.to_datetime(df_PET['date'])
    df_PET['year'] = df_PET['date'].dt.year
    # Group by 'year' and 'month' and calculate the sum of the 'value' column for each month
    annual_sum_ppt = df_PET.groupby(['year'])['value'].sum().reset_index()
    # Filter the years from 2002 to 2019
    selected_years = range(2002, 2020)
    filtered_sum = annual_sum_ppt[annual_sum_ppt['year'].isin(selected_years)]
    # Compute the mean of the monthly sums for the years 2002 to 2019
    mean_PET = filtered_sum['value'].mean()
    # mean_PET = mean_PET * basin_area / 86.4

    RD_sim_obs = []
    RD_TC_PET = []
    for fold in fold_list:
        df_wb = pd.DataFrame(columns=['RD_sim_obs', 'RD_TC_PET'])
        file_name_MCLSTM = '%s_cumQ_%s_%depoch_%dnodes_%.5flr_%sfold.xlsx' % (
            'MCLSTM', basin_name, n_epochs, hidden_nodes_num, learning_rate, fold)
        PATH_MCLSTM = os.path.join(folder_path, file_name_MCLSTM)
        # print('**** file: ', file_name_MCLSTM)
        cols = ['date', 'obs', 'preds', 'TC', 'CellState']
        df_MCLSTM = pd.read_excel(PATH_MCLSTM, names=cols)

        # ============================================#
        # compute the long-term mean                 #
        # ============================================#
        df_MCLSTM['date'] = pd.to_datetime(df_MCLSTM['date'])
        df_MCLSTM['year'] = df_MCLSTM['date'].dt.year
        # Group by 'year' and 'month' and calculate the sum of the 'value' column for each month
        annual_sum_obs = df_MCLSTM.groupby(['year'])['obs'].sum().reset_index()
        annual_sum_preds = df_MCLSTM.groupby(['year'])['preds'].sum().reset_index()
        annual_sum_TC = df_MCLSTM.groupby(['year'])['TC'].sum().reset_index()

        # Filter the years from 2002 to 2019
        selected_years = range(2002, 2020)
        filtered_sum_obs = annual_sum_obs[annual_sum_obs['year'].isin(selected_years)]
        filtered_sum_preds = annual_sum_preds[annual_sum_preds['year'].isin(selected_years)]
        filtered_sum_TC = annual_sum_TC[annual_sum_TC['year'].isin(selected_years)]

        # Compute the mean of the monthly sums for the years 2002 to 2019
        mean_obs = filtered_sum_obs['obs'].mean() * 86.4 / basin_area
        mean_preds = filtered_sum_preds['preds'].mean() * 86.4 / basin_area
        mean_TC = filtered_sum_TC['TC'].mean() * 86.4 / basin_area

        MC_diff_obs = mean_preds - mean_obs
        TC_diff_PET = mean_TC - mean_PET
        print(basin_name, fold, MC_diff_obs / mean_obs * 100, TC_diff_PET / mean_PET * 100)
        RD_sim_obs.append(MC_diff_obs / mean_obs * 100)
        RD_TC_PET.append(TC_diff_PET / mean_PET * 100)

    df_wb['RD_sim_obs'] = np.array(RD_sim_obs)
    df_wb['RD_TC_PET'] = np.array(RD_TC_PET)
    # Compute the mean for each column
    column_means = df_wb.mean()
    # Add the mean values to the last row
    df_wb.loc[len(df_wb)] = column_means
    dfs.append(df_wb)
    if basin_name == 'Delaware':
        basin_name = 'Bushkill'
    basin_names.append(basin_name)

fig, ax = plt.subplots(figsize=(7, 5))
fontsize = 12
font = {'family': 'Arial', 'size': 12}
matplotlib.rc('font', **font)
# Define colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
# Add a vertical line at x=0
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
# Scatter plot for each basin
for i, df in enumerate(dfs):
    # Scatter plot for the first two points with the assigned color
    plt.scatter(df['RD_sim_obs'].iloc[0:3], df['RD_TC_PET'].iloc[0:3], label=None, s=100, color=colors[i])
    # Scatter plot for the last point with a different size (s=100) and the assigned color
    plt.scatter(df['RD_sim_obs'].iloc[-1], df['RD_TC_PET'].iloc[-1], label=basin_names[i], s=200, color=colors[i], edgecolors='black', linewidth=1.5)
# Plot customization
plt.xlabel('Relative bias (%)', fontsize=fontsize)
plt.ylabel('$RD_{TC}$ (%)', fontsize=fontsize)
plt.legend(fontsize=12, scatterpoints=1, markerscale=0.7, loc = 'upper left')
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlim(-40, 40)
# Show the plot
plt.show()
