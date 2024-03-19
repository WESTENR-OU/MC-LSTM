'''
After finishing the training and getting the trained models, run this script first.
This script load the trained models and does two things
(1) Calculate the statistics for each fold and 3-fold mean.
    If saving flag is True, the stats will be saved to defined path.
(2) Generate a time series csv for each watershed.
    If saving flag is True, the time series will be saved to defined path.
'''

import sys
import os
from pathlib import Path

sys.path.append('../')
module_path = os.path.join(Path(__file__).parent, 'mc-lstm-main\mc-lstm-main')
sys.path.append(module_path)
import mclstm_modifiedhydrology
from LEM import LEM_model
from torch.utils.data import DataLoader

sys.path.append('../')
import torch
import load_data
import LSTM_model
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import os
from load_data import compute_metrics

basin_dict = {
    #"Delaware": {"basin_id": "01439500", "basin_area_km2": 306},
    #"Mill": {"basin_id": "06888500", "basin_area_km2": 843},
    #"Sycamore": {"basin_id": "09510200", "basin_area_km2": 428},
    #"Wimberley": {"basin_id": "08104900", "basin_area_km2": 348},
    "Bull": {"basin_id": "06224000", "basin_area_km2": 484},
    #"Sauk": {"basin_id": "12189500", "basin_area_km2": 1849},
    #"Stevens": {"basin_id": "02196000", "basin_area_km2": 1412},
    #"Suwannee": {"basin_id": "02314500", "basin_area_km2": 2926},
    # "Gasconade": {"basin_id": "06934000 ", "basin_area_km2": 8236},
    # "Withlacoochee": {"basin_id": "12189500", "basin_area_km2": 376}
    # "HRU_8013000": {"basin_id": "08013000", "basin_area_km2": 1292.4}
}
save_timeseries = True
save_statistics = True
# folder_path = os.path.join(os.path.dirname(__file__), 'result/timeseries_result/Mill')
# folder_path = os.path.join(os.path.dirname(__file__), 'result/Util/Plottest/Mill')
# folder_path = r"C:\Users\yihan\OneDrive\OU\To_Yihan\To_Yihan\LSTM_test\result\CV_latest_0.001_16_200_seed60\Mill"
folder_path = r"C:\Users\yihan\OneDrive\OU\To_Yihan\To_Yihan\LSTM_test\trained_models_128nodes\LSTM_sensitivity\Sycamore"
files = os.listdir(folder_path)
font = {'family': 'Arial', 'size': 12}
matplotlib.rc('font', **font)
HRU_indicator = True

# Set Seaborn style
sns.set(style="whitegrid")

# plt.figure(figsize=(10, 4))

for file in files:
    print('*** ', file)
    hidden_nodes_cal = None
    cell_state_cal = None
    words = file.split('_')
    model_type = words[0]
    basin_name = words[1]
    import re

    hidden_nodes_num = int(re.search(r'\d+', words[3]).group(0))
    # Extracting the number of epochs
    n_epochs = int(re.search(r'(\d+)epoch', words[2]).group(1))
    # Extracting the learning rate
    learning_rate = float(re.search(r'(\d+\.\d+)lr', words[4]).group(1))
    fold = 'f' + str(int(re.search(r'f(\d+)fold', words[5]).group(1)))

    basin_id = basin_dict[basin_name]["basin_id"]
    basin_area = basin_dict[basin_name]["basin_area_km2"]

    Data_folder = os.path.join(Path(__file__).parent.parent, 'Collaborative_Research\Data')

    ppt_path = Data_folder + "/" + basin_name + ".csv"
    PET_path = Data_folder + "/" + basin_name + "_PET.csv"
    # Obs path
    obs_path = Data_folder + "/" + basin_name + "_Obs.xlsx"

    df_ppt = load_data.load_forcing(ppt_path)
    df_PET = load_data.load_forcing(PET_path)
    df_obs = load_data.load_discharge(obs_path)

    df_list = [df_ppt, df_PET, df_obs]

    # Training set
    batch_size = 64
    if fold == 'f1':
        start_tr_date = pd.Timestamp('2002-01-01')
        end_tr_date = pd.Timestamp('2013-12-31')
    elif fold == 'f2':
        start_tr_date = pd.Timestamp('2007-01-01')
        end_tr_date = pd.Timestamp('2019-12-31')
    elif fold == 'f3':
        start_tr_date1 = pd.Timestamp('2002-01-01')
        end_tr_date1 = pd.Timestamp('2007-12-31')
        start_tr_date2 = pd.Timestamp('2014-01-01')
        end_tr_date2 = pd.Timestamp('2019-12-31')
        start_tr_date = [start_tr_date1, start_tr_date2]
        end_tr_date = [end_tr_date1, end_tr_date2]

    myDataset_train = load_data.WatershedDataset_mclstm(basin_id=basin_id,
                                                        basin_name=basin_name,
                                                        basin_area=basin_area,
                                                        data_list=df_list,  # TODO df_excess_list
                                                        start_time=start_tr_date,
                                                        end_time=end_tr_date,
                                                        fold=fold,
                                                        flag='train')
    cal_loader_ordered = DataLoader(myDataset_train, batch_size=batch_size, shuffle=False)

    # Testing set
    if fold == 'f1':
        start_test_date = pd.Timestamp('2013-01-01')
        end_test_date = pd.Timestamp('2019-12-31')
    elif fold == 'f2':
        start_test_date = pd.Timestamp('2002-01-01')
        end_test_date = pd.Timestamp('2007-12-31')
    elif fold == 'f3':
        start_test_date = pd.Timestamp('2007-01-01')
        end_test_date = pd.Timestamp('2013-12-31')

    myDataset_test = load_data.WatershedDataset_mclstm(basin_id=basin_id,
                                                       basin_name=basin_name,
                                                       basin_area=basin_area,
                                                       data_list=df_list,
                                                       start_time=start_test_date,
                                                       end_time=end_test_date,
                                                       fold=fold)
    test_loader = DataLoader(myDataset_test, batch_size=batch_size, shuffle=False)
    PATH = folder_path + '/' + file
    if model_type == 'LSTM':
        model = LSTM_model.Model(hidden_size=hidden_nodes_num, dropout_rate=0.0)
        model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        # Evaluate on test set
        obs_test, preds_test = LSTM_model.eval_model(model, test_loader)
        obs_cal_ordered, preds_cal_ordered = LSTM_model.eval_model(model, cal_loader_ordered)


    elif model_type == 'MCLSTM':
        mc_lstm_model = mclstm_modifiedhydrology.MassConservingLSTM(1, 1, hidden_nodes_num, time_dependent=False,
                                                                    batch_first=True)
        mc_lstm_model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        # Evaluate on test set
        obs_test, preds_test = mclstm_modifiedhydrology.eval_model(mc_lstm_model, test_loader)
        obs_cal_ordered, preds_cal_ordered = mclstm_modifiedhydrology.eval_model(mc_lstm_model, cal_loader_ordered)

        # trash cell
        obs_cal_ordered, hidden_nodes_cal = \
            mclstm_modifiedhydrology.get_hidden_nodes(mc_lstm_model, cal_loader_ordered)
        obs_test, hidden_nodes_test = \
            mclstm_modifiedhydrology.get_hidden_nodes(mc_lstm_model, test_loader)
        hidden_nodes_cal = hidden_nodes_cal[:, 0] * basin_area / 86.4
        hidden_nodes_test = hidden_nodes_test[:, 0] * basin_area / 86.4

        # cell state
        obs_cal_ordered, cell_state_cal = \
            mclstm_modifiedhydrology.get_cell_states(mc_lstm_model, cal_loader_ordered)
        obs_test, cell_state_test = \
            mclstm_modifiedhydrology.get_cell_states(mc_lstm_model, test_loader)
        cell_state_cal = cell_state_cal * basin_area / 86.4
        cell_state_test = cell_state_test * basin_area / 86.4

    obs_cal_ordered = obs_cal_ordered * basin_area / 86.4
    preds_cal_ordered = preds_cal_ordered * basin_area / 86.4
    obs_test = obs_test * basin_area / 86.4
    preds_test = preds_test * basin_area / 86.4

    seq_length = 365
    date_range = pd.date_range(pd.Timestamp('2002-01-01'), pd.Timestamp('2019-12-31'))
    obs = np.full(date_range.shape[0], None).reshape(-1, 1)
    preds = np.full(date_range.shape[0], None).reshape(-1, 1)
    hidden_nodes = None
    cell_state = None
    if hidden_nodes_cal is not None:
        hidden_nodes = np.full(date_range.shape[0], None).reshape(-1, 1)
    if cell_state_cal is not None:
        cell_state = np.full(date_range.shape[0], None).reshape(-1, 1)
    if fold == 'f1' or fold == 'f2':
        obs[date_range.get_loc(start_tr_date + pd.DateOffset(days=seq_length - 2)) + 1:date_range.get_loc(
            end_tr_date) + 1] = obs_cal_ordered.numpy()
        preds[date_range.get_loc(start_tr_date + pd.DateOffset(days=seq_length - 2)) + 1:date_range.get_loc(
            end_tr_date) + 1] = preds_cal_ordered.numpy().reshape(-1, 1)
        if hidden_nodes_cal is not None:
            hidden_nodes[date_range.get_loc(start_tr_date + pd.DateOffset(days=seq_length - 2)) + 1:date_range.get_loc(
                end_tr_date) + 1] = hidden_nodes_cal.numpy().reshape(-1, 1)
        if cell_state_cal is not None:
            cell_state[date_range.get_loc(start_tr_date + pd.DateOffset(days=seq_length - 2)) + 1:date_range.get_loc(
                end_tr_date) + 1] = cell_state_cal.numpy().reshape(-1, 1)
    elif fold == 'f3':
        a = date_range.get_loc(start_tr_date[0] + pd.DateOffset(days=seq_length - 2)) + 1
        b = date_range.get_loc(end_tr_date[0])
        c = date_range.get_loc(start_tr_date[1])
        d = date_range.get_loc(end_tr_date[1])

        obs[a:b] = obs_cal_ordered.numpy()[:b - a]
        obs[c:d + 1] = obs_cal_ordered.numpy()[b - a + 1:]
        preds[a:b] = preds_cal_ordered.numpy()[:b - a]
        preds[c:d + 1] = preds_cal_ordered.numpy()[b - a + 1:]
        if hidden_nodes_cal is not None:
            hidden_nodes[a:b] = hidden_nodes_cal.numpy()[:b - a].reshape(-1, 1)
            hidden_nodes[c:d + 1] = hidden_nodes_cal.numpy()[b - a + 1:].reshape(-1, 1)
        if cell_state_cal is not None:
            cell_state[a:b] = cell_state_cal.numpy()[:b - a].reshape(-1, 1)
            cell_state[c:d + 1] = cell_state_cal.numpy()[b - a + 1:].reshape(-1, 1)

    obs[date_range.get_loc(start_test_date + pd.DateOffset(days=seq_length - 2)) + 1:date_range.get_loc(
        end_test_date) + 1] = obs_test.numpy()
    preds[date_range.get_loc(start_test_date + pd.DateOffset(days=seq_length - 2)) + 1:date_range.get_loc(
        end_test_date) + 1] = preds_test.numpy()
    if hidden_nodes_cal is not None:
        hidden_nodes[date_range.get_loc(start_test_date + pd.DateOffset(days=seq_length - 2)) + 1:date_range.get_loc(
            end_test_date) + 1] = hidden_nodes_test.numpy().reshape(-1, 1)
    if cell_state_cal is not None:
        cell_state[date_range.get_loc(start_test_date + pd.DateOffset(days=seq_length - 2)) + 1:date_range.get_loc(
            end_test_date) + 1] = cell_state_test.numpy().reshape(-1, 1)

    # Save the results to xlsx
    if hidden_nodes is not None:
        results = pd.DataFrame({
            'Date': date_range,
            'Observed': obs.reshape(-1),
            'Predicted': preds.reshape(-1),
            'HiddenNodes': hidden_nodes.reshape(-1),
            'CellState': cell_state.reshape(-1)
        })
    else:
        results = pd.DataFrame({
            'Date': date_range,
            'Observed': obs.reshape(-1),
            'Predicted': preds.reshape(-1)
        })

    if save_timeseries:
        # Define the file path for the XLSX file
        dir = os.path.dirname(__file__)
        model_name = 'result_128nodes/timeseries_result_LSTM_sens/Sycamore/%s_cumQ_%s_%depoch_%dnodes_%.5flr_%sfold.xlsx' % (
            model_type, basin_name, n_epochs, hidden_nodes_num, learning_rate, fold)
        output_file = os.path.join(dir, model_name)

        # Save the DataFrame to an XLSX file
        results.to_excel(output_file, index=False, header=False)

    ##########################################
    ############## statistics ################
    ##########################################
    cc_cal, rmse_cal, KGE_cal, NSE_cal, cumRB_cal = compute_metrics(obs_cal_ordered.numpy().squeeze(),
                                                                    preds_cal_ordered.numpy().squeeze())
    cc_val, rmse_val, KGE_val, NSE_val, cumRB_val = compute_metrics(obs_test.numpy().squeeze(),
                                                                    preds_test.numpy().squeeze())

    statistics = {
        'Metric': ['CC', 'RMSE', 'KGE', 'NSE', 'cumRB (%)'],
        'Cal': [cc_cal, rmse_cal, KGE_cal, NSE_cal, cumRB_cal],
        'Val': [cc_val, rmse_val, KGE_val, NSE_val, cumRB_val]
    }

    df = pd.DataFrame(statistics)
    if save_statistics:
        dir = os.path.dirname(__file__)
        model_name = 'result_128nodes/statistics_LSTM_sens/Sycamore/3_fold/%s_%s_%depoch_%dnodes_%.5flr_%sfold.csv' % (
            model_type, basin_name, n_epochs, hidden_nodes_num, learning_rate, fold)
        output_file = os.path.join(dir, model_name)
        # Save the DataFrame to a CSV file
        df.to_csv(output_file, index=False)
model_type = 'MCLSTM'
n_epochs = 200
hidden_nodes_num = 128
learning_rates = [0.0002, 0.0003, 0.0004, 0.0006, 0.0007, 0.0008, 0.0009, 0.0012, 0.0015, 0.002, 0.003]
for basin in basin_dict:
    for learning_rate in learning_rates:
        stats_filepath1 = os.path.join(dir,
                                       'result_128nodes/statistics_MC_sens/Bull/3_fold/%s_%s_%depoch_%dnodes_%.5flr_f1fold.csv' % (
                                           model_type, basin, n_epochs, hidden_nodes_num, learning_rate))
        stats_filepath2 = os.path.join(dir,
                                       'result_128nodes/statistics_MC_sens/Bull/3_fold/%s_%s_%depoch_%dnodes_%.5flr_f2fold.csv' % (
                                           model_type, basin, n_epochs, hidden_nodes_num, learning_rate))
        stats_filepath3 = os.path.join(dir,
                                       'result_128nodes/statistics_MC_sens/Bull/3_fold/%s_%s_%depoch_%dnodes_%.5flr_f3fold.csv' % (
                                           model_type, basin, n_epochs, hidden_nodes_num, learning_rate))
        file_paths = [stats_filepath1, stats_filepath2, stats_filepath3]

        # Read each CSV file into a DataFrame and extract relevant columns
        dfs = [pd.read_csv(file_path,skipinitialspace=True) for file_path in file_paths]
        dfs = [df[['Metric', 'Cal', 'Val']] for df in dfs]

        # Concatenate the DataFrames vertically
        merged_df = pd.concat(dfs)

        # Group by "Metric" and calculate the mean
        mean_df = merged_df.groupby('Metric').mean()
        mean_df.to_csv(os.path.join(dir,
                                    'result_128nodes/statistics_MC_sens/Bull/%s_%s_%.5flr.csv' % (
                                        model_type, basin, learning_rate)), index_label='Metric')
