import pandas as pd
import numpy as np
import os
from pathlib import Path
import load_data
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from load_data import compute_metrics
import matplotlib.dates as mdates
import math

basin_dict = {
    "Delaware": {"basin_id": "01439500", "basin_area_km2": 306},
    "Mill": {"basin_id": "06888500", "basin_area_km2": 843},
    "Sycamore": {"basin_id": "09510200", "basin_area_km2": 428},
    "Wimberley": {"basin_id": "08104900", "basin_area_km2": 348},
    "Bull": {"basin_id": "06224000", "basin_area_km2": 484},
    "Sauk": {"basin_id": "12189500", "basin_area_km2": 1849},
    "Stevens": {"basin_id": "02196000", "basin_area_km2": 1412},
    "Suwannee": {"basin_id": "12189500", "basin_area_km2": 2926}
}
font = {'family': 'Arial', 'size': 14}
matplotlib.rc('font', **font)
save_statistics = False
# Create a grid of subplots
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 12), sharex=True)

for i, (basin, basin_info) in enumerate(basin_dict.items()):
    folder_path = os.path.join(Path(__file__).parent, 'result_128nodes/timeseries_result_best/' + basin)
    files = os.listdir(folder_path)
    row_ind = int(i / 2)
    col_ind = math.ceil(i / 2) - row_ind

    #########################################################
    #########################################################
    #################### MC illustration ####################
    #########################################################
    #########################################################
    # initialize the figure

    # plt.figure(figsize=(8, 4))

    # Define some variables
    previous_model_type = None
    obs_cum = None
    preds_cum = None
    diff = None
    diff_absmax = 0
    date_range = pd.date_range(pd.Timestamp('2002-01-01'), pd.Timestamp('2019-12-31'))
    seq_length = 365

    # Draw a 0-line
    axes[row_ind, col_ind].plot(date_range[364:], np.zeros(date_range.shape[0])[364:], alpha=0.7, color='black',
                                linewidth=2,
                                linestyle='--')

    # read LSTM and MCLSTM
    for file in files:
        print("***", file)
        df = pd.read_excel(folder_path + '/' + file, header=None)

        words = file.split('_')
        model_type = words[0]
        basin_name = words[2]

        import re

        hidden_nodes_num = int(re.search(r'\d+', words[4]).group(0))
        # Extracting the number of epochs
        n_epochs = int(re.search(r'(\d+)epoch', words[3]).group(1))
        # Extracting the learning rate
        learning_rate = float(re.search(r'(\d+\.\d+)lr', words[5]).group(1))
        fold = 'f' + str(int(re.search(r'f(\d+)fold', words[6]).group(1)))

        preds = df.fillna(0).values[:, 2]
        obs = df.fillna(0).values[:, 1]

        if fold == 'f1':
            start_tr_date = pd.Timestamp('2002-01-01')
            end_tr_date = pd.Timestamp('2013-12-31')
            start_test_date = pd.Timestamp('2013-01-01')
            end_test_date = pd.Timestamp('2019-12-31')

        elif fold == 'f2':
            start_tr_date = pd.Timestamp('2007-01-01')
            end_tr_date = pd.Timestamp('2019-12-31')
            start_test_date = pd.Timestamp('2002-01-01')
            end_test_date = pd.Timestamp('2007-12-31')

        elif fold == 'f3':
            start_tr_date1 = pd.Timestamp('2002-01-01')
            end_tr_date1 = pd.Timestamp('2007-12-31')
            start_tr_date2 = pd.Timestamp('2014-01-01')
            end_tr_date2 = pd.Timestamp('2019-12-31')
            start_tr_date = [start_tr_date1, start_tr_date2]
            end_tr_date = [end_tr_date1, end_tr_date2]
            start_test_date = pd.Timestamp('2007-01-01')
            end_test_date = pd.Timestamp('2013-12-31')

        if model_type != previous_model_type:
            if diff is not None:
                if previous_model_type == 'LSTM':
                    color = 'firebrick'
                elif previous_model_type == 'MCLSTM':
                    color = 'deepskyblue'
                # if fold == "f1" and previous_model_type == 'LSTM':
                #    print('no')
                #    plt.plot(date_range, obs_cum, label="Observation", color='black', linewidth=2) # TODO
                # plt.plot(date_range, preds_cum, label=previous_model_type + ' Simulation', alpha=0.65, color=color, linewidth=2) # TODO
                axes[row_ind, col_ind].plot(date_range[364:], diff[364:], label=previous_model_type + ' Simulation',
                                            alpha=1, linewidth=2, color=color)

            # Create new obs_test and preds_test for the new model type
            obs_test = np.full(date_range.shape[0], None)
            preds_test = np.full(date_range.shape[0], None)

        # Otherwise, fill the obs_test and preds_test with three folds of current model type
        start_loc = date_range.get_loc(start_test_date + pd.DateOffset(days=seq_length - 1))
        end_loc = date_range.get_loc(end_test_date) + 1

        obs_test[start_loc:end_loc] = obs[start_loc:end_loc]
        preds_test[start_loc:end_loc] = preds[start_loc:end_loc]

        previous_model_type = model_type

        obs_cum = np.cumsum(np.array([0 if x is None else x for x in obs_test]))
        preds_cum = np.cumsum(np.array([0 if x is None else x for x in preds_test]))
        diff = (preds_cum - obs_cum) * 86400  # convert cms to m^3
        if diff_absmax < max(np.abs(diff)):
            diff_absmax = max(np.abs(diff))  # for setting ylim of the plot later

        if previous_model_type == 'LSTM':
            color = 'firebrick'
        elif previous_model_type == 'MCLSTM':
            color = 'deepskyblue'
        if file == files[-1]:
            # plt.plot(date_range, preds_cum, label=previous_model_type + ' Simulation', alpha=0.65, color=color) #TODO
            axes[row_ind, col_ind].plot(date_range[364:], diff[364:], label=previous_model_type + ' Simulation',
                                        alpha=1, color=color,
                                        linewidth=2)

    # Read SAC-SMA
    file_path = os.path.join(Path(__file__).parent.parent, 'SAC_SMA_result/' + basin + '_SacSMA.xlsx')
    df = pd.read_excel(file_path, header=None)
    result_f2 = df.values[:, 0]
    result_f3 = df.values[:, 1]
    result_f1 = df.values[:, 2]

    cum_f1_cal = np.sum(result_f1[364:4383])
    cum_f1_val = np.sum(result_f1[4383:])

    cum_f2_cal = np.sum(result_f2[2191:])
    cum_f2_val = np.sum(result_f2[364:2191])

    cum_f3_cal = np.sum(result_f3[364:2191]) + np.sum(result_f3[4383:])
    cum_f3_val = np.sum(result_f3[2191:4383])

    preds_test_SacSMA = np.full(date_range.shape[0], None)
    preds_test_SacSMA[4383:] = result_f1[4383:]
    preds_test_SacSMA[364:2191] = result_f2[364:2191]
    preds_test_SacSMA[2191:4383] = result_f1[2191:4383]

    preds_SacSMA_cum = np.cumsum(np.array([0 if x is None else x for x in preds_test_SacSMA]))

    diff_SacSMA = (preds_SacSMA_cum - obs_cum) * 86400
    if diff_absmax < max(np.abs(diff_SacSMA)):
        diff_absmax = max(np.abs(diff_SacSMA))  # for setting ylim of the plot later

    axes[row_ind, col_ind].plot(date_range[364:], diff_SacSMA[364:], label='Sac-SMA Simulation', alpha=1, color='olive',
                                linewidth=2)

    '''
    plt.legend(loc='upper left', fontsize='small')

    # Set y-axis limits and label
    ylim_ = diff_absmax * 1.1
    plt.ylim(-ylim_, ylim_)
    plt.ylabel('Cumulative discharge difference ($m^3$)')
    plt.xlabel('Date')

    # Apply scientific notation to y-axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Add grid lines and title
    plt.grid(True, linestyle='--', alpha=0.7)

    # Customize x-axis tick labels for better readability

    plt.xlim([date_range[364], date_range[-1]])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
    # Annotate the initial and last dates on the x-axis
    y_position = -ylim_
    initial_date = date_range[364].strftime('%Y/%m')
    last_date = date_range[-1].strftime('%Y/%m')
    # plt.annotate(initial_date, xy=(date_range[364], y_position), xytext=(0, -40), textcoords='offset points', rotation=45, ha='right', fontsize=plt.rcParams['font.size'])
    plt.annotate(last_date, xy=(date_range[-1], y_position), xytext=(0, -40), textcoords='offset points', rotation=45,
                 ha='right', fontsize=plt.rcParams['font.size'])
    plt.xticks(rotation=45, ha='right')
    if basin == "Delaware":
        basin_name = "Bushkill"
    plt.title(basin_name)

    dir = os.path.dirname(__file__)
    model_name = 'figureset/diff_cumQ_%s_%depoch_%dnodes_%.5flr.png' % (
        basin_name, n_epochs, hidden_nodes_num, learning_rate)
    output_file = os.path.join(dir, model_name)
    save = True
    if save:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    '''

    ylim_ = diff_absmax * 1.1
    # axes[row_ind, col_ind].legend(loc='upper left', fontsize='small')
    axes[row_ind, col_ind].set_ylim(-ylim_, ylim_)
    # axes[row_ind, col_ind].set_ylabel('Cumulative discharge difference ($m^3$)')
    # axes[row_ind, col_ind].set_xlabel('Date')
    axes[row_ind, col_ind].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axes[row_ind, col_ind].grid(True, linestyle='--', alpha=0.7)
    axes[row_ind, col_ind].set_xlim([date_range[364], date_range[-1]])
    axes[row_ind, col_ind].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))

    y_position = -ylim_
    if row_ind == 3:
        step = 365*2  # Adjust 'step' according to your needs
        axes[row_ind, col_ind].set_xticks(date_range[365*2::step])
        axes[row_ind, col_ind].set_xticklabels([date.strftime('%Y/%m') for date in date_range[365*2::step]], rotation=45,
                                               ha='right')
    if basin == "Delaware":
        basin_name = "Bushkill"
    axes[row_ind, col_ind].set_title(basin_name)

# Create a common legend outside the subplots
#handles, labels = axes[0, 0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center', fontsize='small')

dir = os.path.dirname(__file__)
name = 'figureset_128nodes/MC_illustration_best.png'
output_file = os.path.join(dir, name)
save = True
if save:
    plt.savefig(output_file, bbox_inches='tight', dpi=600)

plt.tight_layout()
plt.show()
