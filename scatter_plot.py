import sys
import os
from pathlib import Path

sys.path.append('../')
module_path = os.path.join(Path(__file__).parent, 'mc-lstm-main\mc-lstm-main')
sys.path.append(module_path)

sys.path.append('../')
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import compute_metrics
import math

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
save_highflow_scatterplot = True
save_lowflow_scatterplot = False
save_statistics = False
if __name__ == '__main__':
    model_result = dict()
    for basin in basin_dict:
        basin_name = basin
        print('************ basin %s *************' % {basin_name})
        folder_path = os.path.join(os.path.dirname(__file__), 'result_128nodes/timeseries_result_best/' + basin_name)
        files = os.listdir(folder_path)

        # Define some variables
        previous_model_type = None
        preds_test = None
        date_range = pd.date_range(pd.Timestamp('2002-01-01'), pd.Timestamp('2019-12-31'))
        seq_length = 365

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
                if preds_test is not None:
                    LSTM_preds_test = preds_test
                    LSTM_obs_test = obs_test
                # Create new obs_test and preds_test for the new model type
                obs_test = np.full(date_range.shape[0], None)
                preds_test = np.full(date_range.shape[0], None)

            # Otherwise, fill the obs_test and preds_test with three folds of current model type
            start_loc = date_range.get_loc(start_test_date + pd.DateOffset(days=seq_length - 1))
            end_loc = date_range.get_loc(end_test_date) + 1

            obs_test[start_loc:end_loc] = obs[start_loc:end_loc]
            preds_test[start_loc:end_loc] = preds[start_loc:end_loc]

            previous_model_type = model_type

            if file == files[-1]:
                MCLSTM_preds_test = preds_test
                MCLSTM_obs_test = obs_test

        # Read SAC-SMA
        file_path = os.path.join(Path(__file__).parent.parent, 'SAC_SMA_result/' + basin + '_SacSMA.xlsx')
        df = pd.read_excel(file_path, header=None)
        result_f2 = df.values[:, 0]
        result_f3 = df.values[:, 1]
        result_f1 = df.values[:, 2]

        preds_test_SacSMA = np.full(date_range.shape[0], None)
        preds_test_SacSMA[4383:] = result_f1[4383:]
        preds_test_SacSMA[364:2191] = result_f2[364:2191]
        preds_test_SacSMA[2191:4383] = result_f1[2191:4383]

        model_result[basin_name] = {'obs_test_whole': LSTM_obs_test,
                                    'LSTM_preds_test_whole': LSTM_preds_test,
                                    'MCLSTM_preds_test_whole': MCLSTM_preds_test,
                                    'SacSMA_preds_test_whole': preds_test_SacSMA}

        LSTM = np.array([0 if x is None else x for x in LSTM_obs_test])
        if np.any(LSTM < 0):
            # Count the number of negative elements in the array
            num_negatives = np.sum(LSTM < 0)
            print("!NOTE!:", basin, " has negative")
            print(f"The array contains {num_negatives} negative elements.")

    basin_list = list(basin_dict.items())
    size = 14
    font = {'family': 'Arial', 'size': 14}
    matplotlib.rc('font', **font)

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(6, 6), sharex=True, sharey=True)

    for i in range(8):
        row_ind = int(i / 3)
        col_ind = i - row_ind * 3
        basin_name = basin_list[i][0]
        obs_test_whole = model_result[basin_name]['obs_test_whole'][364:].astype(float)
        LSTM_preds_test_whole = model_result[basin_name]['LSTM_preds_test_whole'][364:].astype(float)
        MCLSTM_preds_test_whole = model_result[basin_name]['MCLSTM_preds_test_whole'][364:].astype(float)
        SacSMA_preds_test_whole = model_result[basin_name]['SacSMA_preds_test_whole'][364:].astype(float)  # TODO

        ##########################################
        ############## high flow #################
        ##########################################
        threshold_high = np.percentile(obs_test_whole, 98)

        # Create a new array containing elements greater than the threshold
        obs_high = np.log10(obs_test_whole[obs_test_whole >= threshold_high])
        print('++++++ threshold 98%: ', basin_name, threshold_high, obs_high.shape[0])
        LSTM_preds_high = np.log10(np.array(
            [LSTM_preds_test_whole[i] for i, obs in enumerate(obs_test_whole) if obs >= threshold_high]))
        MC_preds_high = np.log10(np.array(
            [MCLSTM_preds_test_whole[i] for i, obs in enumerate(obs_test_whole) if obs >= threshold_high]))
        SacSMA_preds_high = np.log10(np.array(
            [SacSMA_preds_test_whole[i] for i, obs in enumerate(obs_test_whole) if obs >= threshold_high]))

        # sns.set(style="whitegrid")
        # plt.figure(figsize=(2.5, 2.5)) todo

        # plt.title(basin_name + '(n=%i)' % obs_high.shape[0], fontsize=12) # todo
        axes[row_ind, col_ind].scatter(obs_high, SacSMA_preds_high, color='olive', alpha=0.6, label='Sac-SMA', s=size)
        axes[row_ind, col_ind].scatter(obs_high, MC_preds_high, color='deepskyblue', alpha=0.5, label='MCLSTM', s=size)
        axes[row_ind, col_ind].scatter(obs_high, LSTM_preds_high, color='firebrick', alpha=0.4, label='LSTM', s=size)

        # Set the same x and y range based on the filtered data for the current plot
        # min_val = min(MC_obs_high.min(), MC_preds_high.min(), obs_high.min(), preds_high.min(), SacSMA_preds_high.min())
        # max_val = max(MC_obs_high.max(), MC_preds_high.max(), obs_high.max(), preds_high.max(), SacSMA_preds_high.max())
        # min_val = min_val * 0.95
        # max_val = max_val*1.05

        min_val = -1
        max_val = 3
        axes[row_ind, col_ind].set_xlim(min_val, max_val)
        axes[row_ind, col_ind].set_ylim(min_val, max_val)

        # plt.xticks(range(min_val, max_val+1))
        # plt.yticks(range(min_val, max_val+1))

        # Specify the tick locations for both the x and y axes
        tick_positions = [-1, 0, 1, 2, 3]

        # Customize the tick labels to be the same
        axes[row_ind, col_ind].set_xticks(tick_positions)
        axes[row_ind, col_ind].set_yticks(tick_positions)

        # Add a 1:1 line
        axes[row_ind, col_ind].plot([min_val, max_val], [min_val, max_val], color='lightgray', linestyle='--')
        # axes[row_ind, col_ind].legend(loc='lower left', fontsize='small')
        if basin_name == "Delaware":
            basin_name = "Bushkill"
        axes[row_ind, col_ind].set_title(basin_name)

    plt.tight_layout()
    fig.text(0.5, 0.01, 'log(Observation)(m$^3$/s)', ha='center')
    fig.text(0.01, 0.5, 'log(Simulation)(m$^3$/s)', va='center', rotation='vertical')

    dir = os.path.dirname(__file__)
    #model_name = 'figureset_128nodes/scatter_plots/highflow_scatterplot_%depoch_%dnodes_%.5flr.png' % (
    #    n_epochs, hidden_nodes_num, learning_rate)
    model_name = 'figureset_128nodes/scatter_plots/highflow_scatterplot_%depoch_%dnodes_best.png' % (
        n_epochs, hidden_nodes_num)
    output_file = os.path.join(dir, model_name)
    if save_highflow_scatterplot:
        plt.savefig(output_file, bbox_inches='tight', dpi=600)

    font = {'family': 'Arial', 'size': 14}
    matplotlib.rc('font', **font)

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(6, 6), sharex=True, sharey=True)
    for i in range(8):
        row_ind = int(i / 3)
        col_ind = i - row_ind * 3
        basin_name = basin_list[i][0]
        obs_test_whole = model_result[basin_name]['obs_test_whole'][364:].astype(float)
        LSTM_preds_test_whole = model_result[basin_name]['LSTM_preds_test_whole'][364:].astype(float)
        MCLSTM_preds_test_whole = model_result[basin_name]['MCLSTM_preds_test_whole'][364:].astype(float)
        SacSMA_preds_test_whole = model_result[basin_name]['SacSMA_preds_test_whole'][364:].astype(float)  # TODO
        ##########################################
        ############## low flow #################
        ##########################################
        # Create a new array containing elements higher than the threshold

        threshold_low = np.percentile(obs_test_whole, 20)
        if threshold_low == 0:
            threshold_low += 0.01
        if threshold_low > 0:
            '''
            # 2023/11/03 #########
            preds_test_whole[preds_test_whole < 0] = 0
            MC_preds_test_whole[MC_preds_test_whole < 0] = 0
            SacSMA_preds_test_whole[SacSMA_preds_test_whole < 0] = 0
            #######################
            '''
            # Create a new array containing elements greater than the threshold
            obs_low = np.log10(obs_test_whole[(obs_test_whole <= threshold_low) & (obs_test_whole > 0.001)])
            print('++++++ threshold 20%: ', basin_name, threshold_low, obs_low.shape[0])
            LSTM_preds_low = np.log10(np.array(
                [LSTM_preds_test_whole[i] for i, obs in enumerate(obs_test_whole) if
                 obs <= threshold_low and obs > 0.001]))
            MC_preds_low = np.log10(np.array(
                [MCLSTM_preds_test_whole[i] for i, obs in enumerate(obs_test_whole) if
                 obs <= threshold_low and obs > 0.001]))
            SacSMA_preds_low = np.log10(np.array(
                [SacSMA_preds_test_whole[i] for i, obs in enumerate(obs_test_whole) if
                 obs <= threshold_low and obs > 0.001]))
            min_val = -3
            max_val = 3
            axes[row_ind, col_ind].set_xlim([min_val, max_val])
            axes[row_ind, col_ind].set_ylim([min_val, max_val])

            axes[row_ind, col_ind].scatter(obs_low, SacSMA_preds_low, color='olive', alpha=0.6, label='Sac-SMA', s=size)
            axes[row_ind, col_ind].scatter(obs_low, MC_preds_low, color='deepskyblue', alpha=0.5, label='MCLSTM',
                                           s=size)
            axes[row_ind, col_ind].scatter(obs_low, LSTM_preds_low, color='firebrick', alpha=0.4, label='LSTM', s=size)

            # Set the same x and y range based on the filtered data for the current plot
            # min_val = min(obs_low.min(), MC_preds_low.min(),LSTM_preds_low.min(), SacSMA_preds_low.min())
            # max_val = max(obs_low.max(), MC_preds_low.max(), LSTM_preds_low.max(), SacSMA_preds_low.max())
            # min_val = min_val * 0.95
            # max_val = max_val*1.05
            # Specify the tick locations for both the x and y axes
            tick_positions = [-3, 0, 3]

            # Customize the tick labels to be the same
            axes[row_ind, col_ind].set_xticks(tick_positions)
            axes[row_ind, col_ind].set_yticks(tick_positions)

            # Add a 1:1 line
            axes[row_ind, col_ind].plot([min_val, max_val], [min_val, max_val], color='lightgray', linestyle='--')
            # axes[row_ind, col_ind].legend(loc='lower left', fontsize='small')
            if basin_name == "Delaware":
                basin_name = "Bushkill"
            axes[row_ind, col_ind].set_title(basin_name)

            ##########################################
            ############## statistics ################
            ##########################################
            cc_lstm, rmse_lstm, KGE_lstm, NSE_lstm, cumRB_lstm = compute_metrics(obs_test_whole, LSTM_preds_test_whole)
            cc_mclstm, rmse_mclstm, KGE_mclstm, NSE_mclstm, cumRB_mclstm = compute_metrics(obs_test_whole,
                                                                                           MCLSTM_preds_test_whole)
            cc_Sac, rmse_Sac, KGE_Sac, NSE_Sac, cumRB_Sac = compute_metrics(obs_test_whole, SacSMA_preds_test_whole)

            statistics = {
                'Metric': ['CC', 'RMSE', 'KGE', 'NSE', 'cumRB (%)'],
                'LSTM': [cc_lstm, rmse_lstm, KGE_lstm, NSE_lstm, cumRB_lstm],
                'MCLSTM': [cc_mclstm, rmse_mclstm, KGE_mclstm, NSE_mclstm, cumRB_mclstm],
                'SacSMA': [cc_Sac, rmse_Sac, KGE_Sac, NSE_Sac, cumRB_Sac]
            }

            df = pd.DataFrame(statistics)
            if save_statistics:
                dir = os.path.dirname(__file__)
                model_name = 'result/statistics/%s.csv' % (basin_name)
                output_file = os.path.join(dir, model_name)
                # Save the DataFrame to a CSV file
                df.to_csv(output_file, index=False)

        plt.tight_layout()
        fig.text(0.5, 0.01, 'log(Observation)(m$^3$/s)', ha='center')
        fig.text(0.01, 0.5, 'log(Simulation)(m$^3$/s)', va='center', rotation='vertical')
        dir = os.path.dirname(__file__)
        #model_name = 'figureset_128nodes/scatter_plots/lowflow_scatterplot_%depoch_%dnodes_%.5flr.png' % (
        #    n_epochs, hidden_nodes_num, learning_rate)
        model_name = 'figureset_128nodes/scatter_plots/lowflow_scatterplot_%depoch_%dnodes_best.png' % (
            n_epochs, hidden_nodes_num)
        output_file = os.path.join(dir, model_name)
        if save_lowflow_scatterplot:
            plt.savefig(output_file, bbox_inches='tight', dpi=360)
