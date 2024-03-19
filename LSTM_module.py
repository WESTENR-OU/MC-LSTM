import copy

def LSTM_experiment(seed, basin_dict, basin_name,
                    lr, n_epochs, hidden_nodes, fold,
                    use_excess=False):
    import sys
    from torch.utils.data import DataLoader
    sys.path.append('../')
    import torch
    import numpy as np
    import pandas as pd
    import tqdm
    import load_data
    import LSTM_model
    import hydroeval
    import plot
    import os
    from pathlib import Path

    print('**** LSTM EXPERIMENT FOR BASIN ', basin_name, " ****")
    print('***** Parameters: [lr = %.4f, n_epochs = %d, hidden_nodes = %d, fold = %s]' % (lr, n_epochs, hidden_nodes, fold))

    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ########################################################################
    ###### LOAD DATA #######################################################
    ########################################################################
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

    # Calibration set
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
                                                        flag='train',
                                                        use_excess=use_excess)

    cal_loader = DataLoader(myDataset_train, batch_size=batch_size, shuffle=True)

    # Calculate the sizes for the train and validation sets
    total_size = len(cal_loader.dataset)
    train_size = int(0.7 * total_size)
    val_size = total_size - train_size
    # Split the DataLoader indices, not the DataLoader itself
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    # Create DataLoader for train_set using SubsetRandomSampler
    tr_loader = DataLoader(
        cal_loader.dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    )
    # Create DataLoader for val_set using SubsetRandomSampler
    val_loader = DataLoader(
        cal_loader.dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    )

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
                                                       fold=fold,
                                                       use_excess=use_excess)
    test_loader = DataLoader(myDataset_test, batch_size=batch_size, shuffle=False)

    ########################################################################
    ###### MODEL TRAINING ##################################################
    ########################################################################
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # This line checks if GPU is available
    print('LSTM training using devide ', DEVICE)
    model = LSTM_model.Model(hidden_size=hidden_nodes, dropout_rate=0.0).to(DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08,
                                    weight_decay=0, momentum=0, centered=False)
    loss_func = LSTM_model.RMSELoss()

    loss_tr_list = []
    RMSE_val_list = []
    RMSE_tr_list = []
    best_tr = np.inf
    best_eval = np.inf

    for i in range(n_epochs):
        loss_ave = LSTM_model.train_epoch(model, optimizer, tr_loader, loss_func, i + 1)
        loss_tr_list.append(loss_ave)

        if i % 1 == 0:
            # Eval on validation set
            obs_val, preds_val = LSTM_model.eval_model(model, val_loader)
            RMSE_val = LSTM_model.calc_rmse(obs_val.numpy(), preds_val.numpy())
            RMSE_val_list.append(RMSE_val)

            # Eval on training set
            obs_tr, preds_tr = LSTM_model.eval_model(model, tr_loader)
            RMSE_tr = LSTM_model.calc_rmse(obs_tr.numpy(), preds_tr.numpy())
            RMSE_tr_list.append(RMSE_tr)

            tqdm.tqdm.write(f"Train and validation RMSE: {loss_ave:.2f}, {RMSE_val:.2f}")

        if RMSE_val < best_eval:
            print('** update')
            best_model = copy.deepcopy(model)
            best_eval = RMSE_val
            best_tr = RMSE_tr

    model = best_model
    # Evaluate on test set
    obs_test, preds_test = LSTM_model.eval_model(model, test_loader)
    obs_cal_ordered, preds_cal_ordered = LSTM_model.eval_model(model, cal_loader_ordered)

    obs_cal_ordered = obs_cal_ordered * basin_area / 86.4
    preds_cal_ordered = preds_cal_ordered * basin_area / 86.4
    obs_test = obs_test * basin_area / 86.4
    preds_test = preds_test * basin_area / 86.4

    RMSE_cal = LSTM_model.calc_rmse(obs_cal_ordered.numpy(),  preds_cal_ordered.numpy())
    RMSE_test = LSTM_model.calc_rmse(obs_test.numpy(), preds_test.numpy())

    # Save the model
    dir = os.path.dirname(__file__)
    model_name = 'trained_models/LSTM/LSTM_%s_%depoch_%dnodes_%.5flr_%sfold' % (
        basin_name, n_epochs, hidden_nodes, lr, fold)
    model_path = os.path.join(dir, model_name)
    torch.save(model.state_dict(), model_path)
    print("Model saved successfully. Path is: ", model_path)

    ########################################################################
    ###### VISUALIZATION ###################################################
    ########################################################################
    plot.plot_RMSE(RMSE_tr_list, RMSE_val_list, basin_name=basin_name,
                   model_type='LSTM', n_epochs=n_epochs, hidden_nodes=hidden_nodes, learning_rate=lr,fold=fold)

    ########################################################################
    ###### MODEL STATISTICS ################################################
    ########################################################################
    KGE_cal = hydroeval.kge(preds_cal_ordered.numpy().squeeze(), obs_cal_ordered.numpy().squeeze())
    KGE_test = hydroeval.kge(preds_test.numpy().squeeze(), obs_test.numpy().squeeze())

    # Create a dictionary with data for each column
    data = {
        'KGE': [KGE_cal[0][0], KGE_test[0][0]],
        'NSE': [hydroeval.nse(preds_cal_ordered.numpy().squeeze(), obs_cal_ordered.numpy().squeeze()),
                # hydroeval.nse(obs_val_ordered.numpy().squeeze(), preds_val_ordered.numpy().squeeze()),
                hydroeval.nse(preds_test.numpy().squeeze(),obs_test.numpy().squeeze())],
        'CC': [np.corrcoef(obs_cal_ordered.numpy().squeeze(), preds_cal_ordered.numpy().squeeze())[0][1],
               # np.corrcoef(obs_val_ordered.numpy().squeeze(), preds_val_ordered.numpy().squeeze())[0][1],
               np.corrcoef(obs_test.numpy().squeeze(), preds_test.numpy().squeeze())[0][1]],
        'RMSE': [RMSE_cal,
                 #best_eval,
                 RMSE_test]
    }

    # Create a DataFrame from the dictionary
    statistics = pd.DataFrame(data)
    print('=== ' + model_path + ' statistics ===')
    print(statistics)


def MCLSTM_experiment(seed, basin_dict, basin_name,
                      lr, n_epochs, hidden_nodes,
                      fold,
                      use_excess=False):
    import sys
    import os
    from pathlib import Path
    sys.path.append('../')
    module_path = os.path.join(Path(__file__).parent, 'mc-lstm-main\mc-lstm-main')
    sys.path.append(module_path)
    from torch.utils.data import DataLoader
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib
    import tqdm
    import load_data
    import mclstm_modifiedhydrology
    import plot
    import hydroeval
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print('**** MC-LSTM EXPERIMENT FOR BASIN ', basin_name, " ****")
    print('***** Parameters: [lr = %.5f, n_epochs = %d, hidden_nodes = %d, fold = %s]' % (lr, n_epochs, hidden_nodes, fold))

    ########################################################################
    ###### LOAD DATA #######################################################
    ########################################################################
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
                                                        flag='train',
                                                        use_excess=use_excess)
    cal_loader = DataLoader(myDataset_train, batch_size=batch_size, shuffle=True)

    # Calculate the sizes for the train and validation sets
    total_size = len(cal_loader.dataset)
    train_size = int(0.7 * total_size)
    val_size = total_size - train_size
    # Split the DataLoader indices, not the DataLoader itself
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    # Create DataLoader for train_set using SubsetRandomSampler
    tr_loader = DataLoader(
        cal_loader.dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    )
    # Create DataLoader for val_set using SubsetRandomSampler
    val_loader = DataLoader(
        cal_loader.dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    )

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
                                                       fold=fold,
                                                       use_excess=use_excess)
    test_loader = DataLoader(myDataset_test, batch_size=batch_size, shuffle=False)

    #tr_loader_ordered = DataLoader(myDataset_train, batch_size=batch_size, shuffle=False)
    #val_loader_ordered = DataLoader(myDataset_val, batch_size=batch_size, shuffle=False)

    ########################################################################
    ###### MODEL TRAINING ##################################################
    ########################################################################
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # This line checks if GPU is available
    mc_lstm_model = mclstm_modifiedhydrology.MassConservingLSTM(1, 1, hidden_nodes, time_dependent=False,
                                                                batch_first=True).to(DEVICE)
    #optimizer = torch.optim.Adam(mc_lstm_model.parameters(), lr=lr)
    optimizer = torch.optim.RMSprop(mc_lstm_model.parameters(), lr=lr, alpha=0.99,
                                    eps=1e-08, weight_decay=0, momentum=0, centered=False)
    loss_func = mclstm_modifiedhydrology.RMSELoss()

    RMSE_tr_list = []
    RMSE_val_list = []
    loss_list = []
    best_tr = np.inf
    best_eval = np.inf

    for i in range(n_epochs):
        loss_ave = mclstm_modifiedhydrology.train_epoch(mc_lstm_model, optimizer, tr_loader, loss_func, i + 1)
        loss_list.append(loss_ave)
        if i % 1 == 0:
            # Eval on validation set
            obs_val, preds_val = mclstm_modifiedhydrology.eval_model(mc_lstm_model, val_loader)
            RMSE_val = mclstm_modifiedhydrology.calc_rmse(obs_val.numpy(), preds_val.numpy())
            RMSE_val_list.append(RMSE_val)

            # Eval on training set
            obs_tr, preds_tr = mclstm_modifiedhydrology.eval_model(mc_lstm_model, tr_loader)
            RMSE_tr = mclstm_modifiedhydrology.calc_rmse(obs_tr.numpy(), preds_tr.numpy())
            RMSE_tr_list.append(RMSE_tr)
            tqdm.tqdm.write(f"Train and validation RMSE: {RMSE_tr:.2f}, {RMSE_val:.2f}")
        if RMSE_val < best_eval:
            print('** update')
            best_model = copy.deepcopy(mc_lstm_model)
            best_eval = RMSE_val
            best_tr = RMSE_tr

    mc_lstm_model = best_model
    # Evaluate on test set
    obs_test, preds_test = mclstm_modifiedhydrology.eval_model(mc_lstm_model, test_loader)
    obs_cal_ordered, preds_cal_ordered = mclstm_modifiedhydrology.eval_model(mc_lstm_model, cal_loader_ordered)

    obs_cal_ordered = obs_cal_ordered * basin_area / 86.4
    preds_cal_ordered = preds_cal_ordered * basin_area / 86.4
    obs_test = obs_test * basin_area / 86.4
    preds_test = preds_test * basin_area / 86.4

    RMSE_cal = mclstm_modifiedhydrology.calc_rmse(obs_cal_ordered.numpy(), preds_cal_ordered.numpy())
    RMSE_test = mclstm_modifiedhydrology.calc_rmse(obs_test.numpy(), preds_test.numpy())


    # Save the model
    dir = os.path.dirname(__file__)
    model_name = 'trained_models/MCLSTM/MCLSTM_%s_%depoch_%dnodes_%.5flr_%sfold' % (
        basin_name, n_epochs, hidden_nodes, lr,fold)
    model_path = os.path.join(dir, model_name)
    torch.save(mc_lstm_model.state_dict(), model_path)
    print("Model saved successfully. Path is: ", model_path)

    ########################################################################
    ###### VISUALIZATION ###################################################
    ########################################################################
    font = {'family': 'Arial', 'size': 12}
    matplotlib.rc('font', **font)
    seq_length = 365
    '''
    plot.plot_time_series(start_tr_date=start_tr_date, end_tr_date=end_tr_date,
                          #start_val_date=start_val_date, end_val_date=end_val_date,
                          start_test_date=start_test_date, end_test_date=end_test_date,
                          obs_cal=obs_cal_ordered, preds_cal=preds_cal_ordered,
                          #obs_val=obs_val_ordered, preds_val=preds_val_ordered,
                          obs_test=obs_test, preds_test=preds_test, seq_length=seq_length, basin_name=basin_name,
                          model_type='MCLSTM', n_epochs=n_epochs, hidden_nodes=hidden_nodes, learning_rate=lr,fold=fold)
    '''
    plot.plot_RMSE(RMSE_tr_list, RMSE_val_list, basin_name=basin_name, model_type='MCLSTM',
                   n_epochs=n_epochs, hidden_nodes=hidden_nodes, learning_rate=lr,fold=fold)

    ########################################################################
    ###### MODEL STATISTICS ################################################
    ########################################################################
    KGE_cal = hydroeval.kge(preds_cal_ordered.numpy().squeeze(), obs_cal_ordered.numpy().squeeze())
    KGE_test = hydroeval.kge(preds_test.numpy().squeeze(), obs_test.numpy().squeeze())

    # Create a dictionary with data for each column
    data = {
        'KGE': [KGE_cal[0][0], KGE_test[0][0]],
        'NSE': [hydroeval.nse(preds_cal_ordered.numpy().squeeze(), obs_cal_ordered.numpy().squeeze()),
                # hydroeval.nse(obs_val_ordered.numpy().squeeze(), preds_val_ordered.numpy().squeeze()),
                hydroeval.nse(preds_test.numpy().squeeze(),obs_test.numpy().squeeze())],
        'CC': [np.corrcoef(obs_cal_ordered.numpy().squeeze(), preds_cal_ordered.numpy().squeeze())[0][1],
               # np.corrcoef(obs_val_ordered.numpy().squeeze(), preds_val_ordered.numpy().squeeze())[0][1],
               np.corrcoef(obs_test.numpy().squeeze(), preds_test.numpy().squeeze())[0][1]],
        'RMSE': [RMSE_cal,
                 #best_eval,
                 RMSE_test]
    }

    # Create a DataFrame from the dictionary
    statistics = pd.DataFrame(data)
    print('=== ' + model_path + ' statistics ===')
    print(statistics)
    print('---------------------------------------')

def LEM_experiment(seed, basin_dict, basin_name, lr, n_epochs, hidden_nodes, fold,
                   use_excess=False):
    from LEM import LEM_model
    from torch.utils.data import DataLoader, TensorDataset
    import matplotlib
    import os
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    import tqdm
    import load_data
    import seaborn as sns
    import hydroeval

    import plot
    import hydroeval
    from pathlib import Path

    print('**** LEM EXPERIMENT FOR BASIN ', basin_name, " ****")
    print('***** Parameters: [lr = %.5f, n_epochs = %d, hidden_nodes = %d]' % (lr, n_epochs, hidden_nodes))
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ########################################################################
    ###### LOAD DATA #######################################################
    ########################################################################
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
                                                        flag='train',
                                                        use_excess=use_excess)
    cal_loader = DataLoader(myDataset_train, batch_size=batch_size, shuffle=True)

    # Calculate the sizes for the train and validation sets
    total_size = len(cal_loader.dataset)
    train_size = int(0.7 * total_size)
    val_size = total_size - train_size
    # Split the DataLoader indices, not the DataLoader itself
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    # Create DataLoader for train_set using SubsetRandomSampler
    tr_loader = DataLoader(
        cal_loader.dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    )
    # Create DataLoader for val_set using SubsetRandomSampler
    val_loader = DataLoader(
        cal_loader.dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    )

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
                                                       fold=fold,
                                                       use_excess=use_excess)
    test_loader = DataLoader(myDataset_test, batch_size=batch_size, shuffle=False)

    ########################################################################
    ###### MODEL TRAINING ##################################################
    ########################################################################
    model = LEM_model.LEM(2, hidden_nodes, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = LEM_model.RMSELoss()

    loss_tr_list = []
    RMSE_tr_list = []
    RMSE_val_list = []

    best_eval = np.inf

    for i in range(n_epochs):
        loss_ave = LEM_model.train_epoch(model, optimizer, tr_loader, loss_func, i + 1)
        loss_tr_list.append(loss_ave)
        if i % 1 == 0:
            # Eval on validation set
            obs_val, preds_val = LEM_model.eval_model(model, val_loader)
            RMSE_val = LEM_model.calc_rmse(obs_val.numpy(), preds_val.numpy())
            RMSE_val_list.append(RMSE_val)

            # Eval on training set
            obs_tr, preds_tr = LEM_model.eval_model(model, tr_loader)
            RMSE_tr = LEM_model.calc_rmse(obs_tr.numpy(), preds_tr.numpy())
            RMSE_tr_list.append(RMSE_tr)

            tqdm.tqdm.write(f"Train and validation RMSE: {RMSE_tr:.2f}, {RMSE_val:.2f}")

        if RMSE_val < best_eval:
            print('update')
            best_model = copy.deepcopy(model)
            best_eval = RMSE_val
    model = best_model
    # Evaluate on test set
    obs_test, preds_test = LEM_model.eval_model(model, test_loader)
    obs_cal_ordered, preds_cal_ordered = LEM_model.eval_model(model, cal_loader_ordered)

    obs_cal_ordered = obs_cal_ordered * basin_area / 86.4
    preds_cal_ordered = preds_cal_ordered * basin_area / 86.4
    obs_test = obs_test * basin_area / 86.4
    preds_test = preds_test * basin_area / 86.4

    RMSE_cal = LEM_model.calc_rmse(obs_cal_ordered.numpy(), preds_cal_ordered.numpy())
    RMSE_test = LEM_model.calc_rmse(obs_test.numpy(), preds_test.numpy())

    # Save the model
    dir = os.path.dirname(__file__)
    model_name = 'result/CV_LEM_0.0005_64_200_seed1/LEM_%s_%depoch_%dnodes_%.5flr_%sfold' % (
        basin_name, n_epochs, hidden_nodes, lr, fold)
    model_path = os.path.join(dir, model_name)
    torch.save(model.state_dict(), model_path)
    print("Model saved successfully. Path is: ", model_path)

    ########################################################################
    ###### VISUALIZATION ###################################################
    ########################################################################
    font = {'family': 'Arial', 'size': 12}
    matplotlib.rc('font', **font)
    seq_length = 365
    plot.plot_time_series(start_tr_date=start_tr_date, end_tr_date=end_tr_date,
                          start_test_date=start_test_date, end_test_date=end_test_date,
                          obs_cal=obs_cal_ordered, preds_cal=preds_cal_ordered,
                          obs_test=obs_test, preds_test=preds_test, seq_length=seq_length, basin_name=basin_name,
                          model_type='LEM', n_epochs=n_epochs, hidden_nodes=hidden_nodes, learning_rate=lr,fold=fold)
    plot.plot_RMSE(RMSE_tr_list, RMSE_val_list, basin_name=basin_name, model_type='LEM',
                   n_epochs=n_epochs, hidden_nodes=hidden_nodes, learning_rate=lr,fold=fold)

    ########################################################################
    ###### MODEL STATISTICS ################################################
    ########################################################################
    KGE_cal = hydroeval.kge(preds_cal_ordered.numpy().squeeze(), obs_cal_ordered.numpy().squeeze())
    KGE_test = hydroeval.kge(preds_test.numpy().squeeze(), obs_test.numpy().squeeze())

    # Create a dictionary with data for each column
    data = {
        'KGE': [KGE_cal[0][0], KGE_test[0][0]],
        'NSE': [hydroeval.nse(preds_cal_ordered.numpy().squeeze(), obs_cal_ordered.numpy().squeeze()),
                # hydroeval.nse(obs_val_ordered.numpy().squeeze(), preds_val_ordered.numpy().squeeze()),
                hydroeval.nse(preds_test.numpy().squeeze(),obs_test.numpy().squeeze())],
        'CC': [np.corrcoef(obs_cal_ordered.numpy().squeeze(), preds_cal_ordered.numpy().squeeze())[0][1],
               # np.corrcoef(obs_val_ordered.numpy().squeeze(), preds_val_ordered.numpy().squeeze())[0][1],
               np.corrcoef(obs_test.numpy().squeeze(), preds_test.numpy().squeeze())[0][1]],
        'RMSE': [RMSE_cal,
                 #best_eval,
                 RMSE_test]
    }
    # Create a DataFrame from the dictionary
    statistics = pd.DataFrame(data)
    print('=== ' + model_path + ' statistics ===')
    print(statistics)
    print('---------------------------------------')
