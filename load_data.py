# Imports
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.stats import pearsonr
import hydroeval


def load_forcing(file_path: str):
    # load the meteorological forcing data of a basin
    # In SC-SAHEL, the forcings are precipitation and potential evapotranspiration
    # file_path: path of the forcing file of a basin as a string
    # return: pd.DataFrame containing the meteorological forcing data

    # read-in data and convert date to datetime index
    df = pd.read_csv(file_path,
                     names=["date", "value"])
    '''
    # Remove the header (first row)
    df = df.iloc[1:]
    # Reset the index if needed
    df.reset_index(drop=True, inplace=True)
    '''
    df.date = pd.to_datetime(df.date, format="%m/%d/%Y")
    return df


def load_forcing_HRU(file_path: str):
    # load the meteorological forcing data of a basin
    # In SC-SAHEL, the forcings are precipitation and potential evapotranspiration
    # file_path: path of the forcing file of a basin as a string
    # return: pd.DataFrame containing the meteorological forcing data

    # read-in data and convert date to datetime index
    df = pd.read_csv(file_path,
                     names=["date", "value"])
    df.date = pd.to_datetime(df.date, format="%m/%d/%Y")
    return df


def load_discharge(file_path: str):
    # load the discharge data of a basin
    # file_path: path of the forcing file of a basin as a string
    # return: pd.DataFrame containing the meteorological forcing data

    # The discharge raw data follows this format:
    # index; site_no; Date; Flow(cfs); Flow(cms)
    df = pd.read_excel(file_path,
                       names=["index", "site_no", "date", "Flow(cfs)", "Flow(cms)"])
    df.date = pd.to_datetime(df.date, format="%m/%d/%Y")
    return df


def load_discharge_HRU(file_path: str):
    df = pd.read_excel(file_path, header=None,
                       names=["site_no", "year", "month", "day", "Flow(cfs)", "dummy"])
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    return df


def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction

    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        #x_new[i, :, :num_features] = x[i:i + seq_length, :]
        x_new[i, :, :] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1]

    return x_new, y_new


class WatershedDataset(Dataset):
    """Torch Dataset for basic use of data of the watersheds

    This data set provides meteorological observations
    (i.e., precipitation and potential evapotranspiration) and discharge of a given
    basin.
    """

    def __init__(self, basin_id: str, basin_name: str, basin_area: float, data_list: list,
                 start_time: pd.Timestamp, end_time: pd.Timestamp, seq_length: int = 365):
        """Initialize Dataset containing the fdata of a single basin.
        :param basin_id: 8-digit USGS Gauge ID of basin as string.
        :param basin_name: name of the basin as string
        :param data_list: A list of data frame column that contains the forcings
        :param start_time & end_time: pandas Timestamp in %Y-%m-%d
        """
        self.basin_id = basin_id
        self.basin_name = basin_name
        self.basin_area = basin_area
        self.data_list = data_list
        self.seq_length = seq_length
        self.means = []  # mean value list of the original input forcing data (for rescaling purpose)
        self.stds = []  # std value list of the original input forcing data (for rescaling purpose)
        self.start_time = start_time
        self.end_time = end_time
        self.features = None
        self.target = None
        self.x = None
        self.y = None

    def preprocess(self):
        df_normalized, df_obs = self.slice_and_normalize_data(forcingdata_list=self.data_list[:-1],
                                                              obsdata=self.data_list[-1])
        # df_normalized.append(df_obs["Flow(cms)"])
        df_normalized.append(df_obs['Flow(cms)'] * 86.4 / self.basin_area)

        # reshape df_normalized from a list to an array (3652, 2)
        # PPT and PET, and Flow Obs
        PPT_value = df_normalized[0].values  # They are all arrays
        PET_value = df_normalized[1].values
        Flow_value = df_normalized[2].values
        forcing_array = np.hstack((PPT_value, PET_value))
        self.x, self.y = reshape_data(forcing_array, Flow_value, self.seq_length)

    def __len__(self):
        self.preprocess()
        return len(self.x)

    def __getitem__(self, idx: int):
        # print(f"Batch index: {idx // 512}, Item index: {idx % 512}")
        # convert arrays to torch tensors
        self.features = torch.from_numpy(self.x.astype(np.float32))
        self.target = torch.from_numpy(self.y.astype(np.float32))
        return self.features[idx], self.target[idx]

    def normalize_data(self, df_col: pd.DataFrame):
        # normalize the data
        # df_col: the value col of a data frame that will be normalized/standardized
        # return: normalized/standardized df col; mean value of raw inputs; std of raw inputs
        mean = df_col.mean()
        std = df_col.std()
        return (df_col - mean) / std, mean, std

    def slice_and_normalize_data(self, forcingdata_list: list, obsdata: pd.DataFrame):
        # Slice the dataset given a time range and normalize the forcing data.
        # Obs data only does slicing.
        # forcingdata_list: A list of df column where each element in the list is a forcing dataset
        # obsdata: a df column
        # return: sliced and normalized forcing data list, and sliced obs data.
        df_obs = obsdata.loc[(obsdata['date'] >= self.start_time) & (obsdata['date'] <= self.end_time)]
        df_selected_list = []
        df_normalized = []
        for df in forcingdata_list:
            df_selected = df.loc[(df['date'] >= self.start_time) & (df['date'] <= self.end_time)]
            df_selected_list.append(df_selected)
            value = df_selected.loc[:, df_selected.columns.isin(['value'])]
            normalized_temp, mean, std = self.normalize_data(value)
            self.means.append(mean)
            self.stds.append(std)
            df_normalized.append(normalized_temp)
        return df_normalized, df_obs

    def rescale_back(self):
        rescaled_features = []
        for i in range(len(self.features)):
            rescaled_features.append(np.array(self.features[i]) * np.array(self.stds[i]) + np.array(self.means[i]))
        return rescaled_features  # nparray


class WatershedDataset_mclstm(Dataset):
    """Torch Dataset for basic use of data of the watersheds

    This data set provides meteorological observations
    (i.e., precipitation and potential evapotranspiration) and discharge of a given
    basin.
    """

    def __init__(self, basin_id: str, basin_name: str, basin_area: float, data_list: list,
                 start_time, end_time, fold: str, use_excess: bool = False, flag: str = 'none',
                 seq_length: int = 365):
        """Initialize Dataset containing the fdata of a single basin.
        :param basin_id: 8-digit USGS Gauge ID of basin as string.
        :param basin_name: name of the basin as string
        :param basin_area: area of the basin (km2)
        :param data_list: A list of data frame column that contains the forcings
        :param start_time & end_time: pandas Timestamp in %Y-%m-%d
        """
        self.x = None
        self.y = None
        self.basin_id = basin_id
        self.basin_name = basin_name
        self.basin_area = basin_area
        self.data_list = data_list
        self.seq_length = seq_length
        self.means = []  # mean value list of the original input forcing data (for rescaling purpose)
        self.stds = []  # std value list of the original input forcing data (for rescaling purpose)
        self.start_time = start_time
        self.end_time = end_time
        self.features = None
        self.target = None
        self.use_excess = use_excess
        self.fold = fold
        self.flag = flag

    def preprocess(self):
        # get reshaped features and targets
        if self.fold == 'f3' and self.flag == 'train':
            df_forcing, df_obs = self.slice_data_fold3cal(forcingdata_list=self.data_list[:-1],
                                                 obsdata=self.data_list[-1])
        else:
            df_forcing, df_obs = self.slice_data(forcingdata_list=self.data_list[:-1],
                                             obsdata=self.data_list[-1])
        if not self.use_excess:
            df_forcing.append(df_obs["Flow(cfs)"] * 0.0283168 * 86.4 / self.basin_area) #mm/day
        else:
            df_forcing.append(df_obs["excess"])
        # reshape df_normalized from a list to an array (3652, 2)
        # PPT and PET, and Flow Obs
        PPT_value = df_forcing[0].values[:, 1].reshape(-1, 1)  # They are all arrays
        PET_value = df_forcing[1].values[:, 1].reshape(-1, 1)
        Flow_value = df_forcing[2].values.reshape(-1, 1)
        forcing_array = np.hstack((PPT_value, PET_value))

        self.x, self.y = reshape_data(forcing_array, Flow_value, self.seq_length)

    def __len__(self):
        self.preprocess()
        return len(self.x)

    def __getitem__(self, idx: int):
        # print(f"Batch index: {idx // 512}, Item index: {idx % 512}")
        # convert arrays to torch tensors
        self.features = torch.from_numpy(self.x.astype(np.float32))
        self.target = torch.from_numpy(self.y.astype(np.float32))
        return self.features[idx], self.target[idx]

    def slice_data(self, forcingdata_list: list, obsdata: pd.DataFrame):
        df_obs = obsdata.loc[(obsdata['date'] >= self.start_time) & (obsdata['date'] <= self.end_time)]
        df_selected_list = []

        for df in forcingdata_list:
            df_selected = df.loc[(df['date'] >= self.start_time) & (df['date'] <= self.end_time)]
            df_selected_list.append(df_selected)

        return df_selected_list, df_obs

    def slice_data_fold3cal(self, forcingdata_list: list, obsdata: pd.DataFrame):
        df_obs_part1 = obsdata.loc[(obsdata['date'] >= self.start_time[0]) & (obsdata['date'] <= self.end_time[0])]
        df_obs_part2 = obsdata.loc[(obsdata['date'] >= self.start_time[1]) & (obsdata['date'] <= self.end_time[1])]
        df_obs = pd.concat([df_obs_part1, df_obs_part2])
        df_obs.reset_index(drop=True, inplace=True)
        df_selected_list = []
        for df in forcingdata_list:
            df_selected_part1 = df.loc[(df['date'] >= self.start_time[0]) & (df['date'] <= self.end_time[0])]
            df_selected_part2 = df.loc[(df['date'] >= self.start_time[1]) & (df['date'] <= self.end_time[1])]
            df_selected = pd.concat([df_selected_part1, df_selected_part2])
            df_selected.reset_index(drop=True, inplace=True)
            df_selected_list.append(df_selected)
        return df_selected_list, df_obs

    def rescale_back(self):
        rescaled_features = []
        for i in range(len(self.features)):
            rescaled_features.append(np.array(self.features[i]) * np.array(self.stds[i]) + np.array(self.means[i]))
        return rescaled_features  # nparray


class WatershedDataset_HRU(Dataset):
    """Torch Dataset for basic use of data of the watersheds

    This data set provides meteorological observations
    (i.e., precipitation and potential evapotranspiration) and discharge of a given
    basin.
    """

    def __init__(self, basin_id: str, basin_name: str, data_list: list,
                 start_time: pd.Timestamp, end_time: pd.Timestamp, seq_length: int = 365):
        """Initialize Dataset containing the fdata of a single basin.
        :param basin_id: 8-digit USGS Gauge ID of basin as string.
        :param basin_name: name of the basin as string
        :param data_list: A list of data frame column that contains the forcings
        :param start_time & end_time: pandas Timestamp in %Y-%m-%d
        """
        self.basin_id = basin_id
        self.basin_name = basin_name
        self.data_list = data_list
        self.seq_length = seq_length
        self.means = []  # mean value list of the original input forcing data (for rescaling purpose)
        self.stds = []  # std value list of the original input forcing data (for rescaling purpose)
        self.start_time = start_time
        self.end_time = end_time
        self.features = None
        self.target = None

    def __len__(self):
        df_normalized, df_obs = self.slice_and_normalize_data(forcingdata_list=self.data_list[:-1],
                                                              obsdata=self.data_list[-1])
        df_normalized.append(df_obs['Flow(cfs)'] * 0.0283)
        # reshape df_normalized from a list to an array (3652, 2)
        # PPT and PET, and Flow Obs
        PPT_value = df_normalized[0].values  # They are all arrays
        PET_value = df_normalized[1].values
        Flow_value = df_normalized[2].values
        forcing_array = np.hstack((PPT_value, PET_value))
        x, y = reshape_data(forcing_array, Flow_value, self.seq_length)

        return len(x)

    def __getitem__(self, idx: int):
        # print(f"Batch index: {idx // 512}, Item index: {idx % 512}")
        df_normalized, df_obs = self.slice_and_normalize_data(forcingdata_list=self.data_list[:-1],
                                                              obsdata=self.data_list[-1])
        # df_normalized: a list of df col. [3652 rows * 1 cols, 3652 rows * 1 cols]
        # df_obs: df. 3652 rows * 5 cols

        df_normalized.append(df_obs["Flow(cfs)"] * 0.0283)
        # reshape df_normalized from a list to an array (3652, 2)
        # PPT and PET, and Flow Obs
        PPT_value = df_normalized[0].values  # They are all arrays
        PET_value = df_normalized[1].values
        Flow_value = df_normalized[2].values
        forcing_array = np.hstack((PPT_value, PET_value))
        x, y = reshape_data(forcing_array, Flow_value, self.seq_length)

        # convert arrays to torch tensors
        self.features = torch.from_numpy(x.astype(np.float32))
        self.target = torch.from_numpy(y.astype(np.float32))
        return self.features[idx], self.target[idx]

    def normalize_data(self, df_col: pd.DataFrame):
        # normalize the data
        # df_col: the value col of a data frame that will be normalized/standardized
        # return: normalized/standardized df col; mean value of raw inputs; std of raw inputs
        mean = df_col.mean()
        std = df_col.std()
        return (df_col - mean) / std, mean, std

    def slice_and_normalize_data(self, forcingdata_list: list, obsdata: pd.DataFrame):
        # Slice the dataset given a time range and normalize the forcing data.
        # Obs data only does slicing.
        # forcingdata_list: A list of df column where each element in the list is a forcing dataset
        # obsdata: a df column
        # return: sliced and normalized forcing data list, and sliced obs data.
        df_obs = obsdata.loc[(obsdata['date'] >= self.start_time) & (obsdata['date'] <= self.end_time)]
        df_selected_list = []
        df_normalized = []
        for df in forcingdata_list:
            df_selected = df.loc[(df['date'] >= self.start_time) & (df['date'] <= self.end_time)]
            df_selected_list.append(df_selected)
            value = df_selected.loc[:, df_selected.columns.isin(['value'])]
            normalized_temp, mean, std = self.normalize_data(value)
            self.means.append(mean)
            self.stds.append(std)
            df_normalized.append(normalized_temp)
        return df_normalized, df_obs

    def rescale_back(self):
        rescaled_features = []
        for i in range(len(self.features)):
            rescaled_features.append(np.array(self.features[i]) * np.array(self.stds[i]) + np.array(self.means[i]))
        return rescaled_features  # nparray


def compute_metrics(observed, predicted):
    # observed and predicted are both np array
    # Compute correlation coefficient
    cc = np.corrcoef(observed, predicted)[0][1]
    # Compute RMSE
    rmse = np.sqrt(np.mean((observed - predicted)**2))
    # Compute KGE
    KGE = hydroeval.kge(predicted, observed)[0][0]
    # Compute NSE
    NSE = hydroeval.nse(predicted, observed)
    # accumulative streamflow bias
    predicted_cum = np.cumsum(np.array([0 if x is None else x for x in predicted]))[-1]
    obs_cum = np.cumsum(np.array([0 if x is None else x for x in observed]))[-1]
    cum_RB = (predicted_cum - obs_cum)/obs_cum*100
    return cc, rmse, KGE, NSE, cum_RB