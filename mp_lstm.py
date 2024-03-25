from multiprocessing import Pool
from LSTM_module_HRU import LSTM_experiment as LSTM_experiment_HRU
from LSTM_module import LSTM_experiment


import time
import torch
import numpy as np

if __name__ == '__main__':
    basin_dict = {
        "Delaware": {"basin_id": "01439500", "basin_area_km2": 306},
        "Mill": {"basin_id": "06888500", "basin_area_km2": 843},
        "Sycamore": {"basin_id": "09510200", "basin_area_km2": 428},
        "Wimberley": {"basin_id": "08104900", "basin_area_km2": 348},
        "Bull": {"basin_id": "06224000", "basin_area_km2": 484},
        "Sauk": {"basin_id": "12189500", "basin_area_km2": 1849},
        "Stevens": {"basin_id": "02196000", "basin_area_km2": 1412},
        "Suwannee": {"basin_id": "02314500", "basin_area_km2": 2926},
    }
    inputs = []
    learning_rate = [0.001]
    n_epochs = [200]
    hidden_nodes = [64]
    folds = ['f1','f2','f3']
    seed = 1
    for basin in basin_dict:
        for lr in learning_rate:
            for epoch in n_epochs:
                for h in hidden_nodes:
                    for fold in folds:
                        inputs.append((seed, basin_dict, basin, lr, epoch, h, fold))
    st = time.time()

    with Pool(processes=1) as p: # change the number of processors. Here processes=1. 
        # can do either HRU experiment and not-HRU experiment
        LSTM_results = p.starmap(LSTM_experiment, inputs)
    et = time.time()
    print('***** Time Consuming: ', (et - st) / 60, "min *****")