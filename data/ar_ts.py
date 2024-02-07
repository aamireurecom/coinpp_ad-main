import random
import torch
import numpy as np
import pprint
import pickle
from torch.utils.data import Dataset
import pathlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA


class Plus1Div2Scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        pass
    def transform(self, X, y=None):
        X = (X + 1) / 2
        return X
    def inverse_transform(self, X, y=None):
        X = X * 2 - 1
        return X


def generate_data(nsample=5000,
                  ar_param_list=(0.2, 0.4),
                  ma_param=1.,
                  ar_param_list_test=(0.02, 0.04),
                  ma_param_test=0.4,
                  len_anomaly=30,
                  num_anomaly=7,
                  seed=0, split='train'):
    np.random.seed(seed)

    list_sim_data = []
    labels = np.zeros((nsample, len(ar_param_list)))
    # Plot 1: AR parameter = +0.9
    for i, ar_param in enumerate(ar_param_list):
        ar1 = np.array([1, -ar_param])
        ma1 = np.array([ma_param])
        AR = ArmaProcess(ar1, ma1)
        list_sim_data.append(np.atleast_2d(AR.generate_sample(nsample=nsample, scale=0.2)).transpose())
    # plt.plot(simulated_data_1)

    if split == 'test':
        list_anomaly_ts = []
        for i, ar_param in enumerate(ar_param_list_test):
            ar1 = np.array([1, -ar_param])
            ma1 = np.array([ma_param_test])
            AR = ArmaProcess(ar1, ma1)
            list_anomaly_ts.append(np.atleast_2d(AR.generate_sample(nsample=nsample, scale=0.2)).transpose())

            for j in range(num_anomaly):
                start = random.randint(0, nsample - len_anomaly)
                list_sim_data[i][start:start+len_anomaly, 0] = list_anomaly_ts[i][start:start+len_anomaly, 0]
                labels[start:start+len_anomaly, i] = 1

    return np.concatenate(list_sim_data, axis=1), labels

def generate_data_SARIMA(nsample=5000,
                  ar_param_list=(0.2, 0.4),
                  ma_param=1.,
                  ar_param_list_test=(0.02, 0.04),
                  ma_param_test=0.4,
                  len_anomaly=30,
                  num_anomaly=7,
                  seed=0, split='train'):
    np.random.seed(seed)

    list_sim_data = []
    labels = np.zeros((nsample, len(ar_param_list)))
    # Plot 1: AR parameter = +0.9
    for i, ar_param in enumerate(ar_param_list):
        ar1 = np.array([1, -ar_param])
        ma1 = np.array([ma_param])
        AR = ARIMA(order=(ar1, ma1))
        list_sim_data.append(np.atleast_2d(AR.generate_sample(nsample=nsample, scale=0.2)).transpose())
    # plt.plot(simulated_data_1)

    if split == 'test':
        list_anomaly_ts = []
        for i, ar_param in enumerate(ar_param_list_test):
            ar1 = np.array([1, -ar_param])
            ma1 = np.array([ma_param_test])
            AR = ArmaProcess(ar1, ma1)
            list_anomaly_ts.append(np.atleast_2d(AR.generate_sample(nsample=nsample, scale=0.2)).transpose())

            for j in range(num_anomaly):
                start = random.randint(0, nsample - len_anomaly)
                list_sim_data[i][start:start+len_anomaly, 0] = list_anomaly_ts[i][start:start+len_anomaly, 0]
                labels[start:start+len_anomaly, i] = 1

    return np.concatenate(list_sim_data, axis=1), labels

class ARTimeSeries(Dataset):
    """SMD dataset without labels.
    Inspired by the coinpp class of LIBRISPEECH dataset.
    Inspired by the Smd_entity class https://github.com/astha-chem/mvts-ano-eval/blob/main/src/datasets/smd_entity.py


    Args:
        patch_shape (int): Shape of patch to use. If -1, uses all data (no patching).
        num_secs (float): Number of seconds of audio to use. If -1, uses all available
            audio.
        normalize (bool): Whether to normalize data to lie in [0, 1].
    """

    def __init__(
        self,
        window_length: int = -1,
        patch_shape: int = -1,
        train_proportion: float = 0.85,
        split: str = 'train',
        normalization_kind: 'str' = 'plus1div2',
        selected_features = (0,),
        ar_param_list = (0.9, 0.3),
        nsample = 1000,
        ma_param=1.,
        *args,
        **kwargs
    ):
        super().__init__() #*args, **kwargs)


        self.split = split
        self.selected_features = np.array(selected_features)


        self.normalization_kind = normalization_kind

        if split == 'train' or split == 'val':
            self.data, labels = generate_data(nsample=nsample,
                                  ar_param_list=ar_param_list,
                                  ma_param=ma_param,
                                  seed=1234)
        elif split == 'test':
            self.data, labels = generate_data(
                nsample=nsample,
                ar_param_list=ar_param_list,
                ma_param=ma_param,
                seed=1235,
                split='test')

        else:
            raise ValueError(f'Invalid value of split {split}. Select one of ("train", "val", "test")')

        if self.normalization_kind == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.normalization_kind == 'standard':
            self.scaler = StandardScaler()
        elif self.normalization_kind == 'plus1div2':
            self.scaler = Plus1Div2Scaler()
        elif self.normalization_kind is None:
            self.scaler = None
        else:
            raise ValueError(f'Invalid value of normalization_kind {normalization_kind}. '
                             f'Select one of ("minmax", "standard", "plus1div2", None)')

        # Take the first half of the train file as training data
        if split == 'train':
            self.data = self.data[:int(train_proportion*len(self.data))]
            self.patch_shape = patch_shape
            self.random_crop = patch_shape != -1
            self.labels = labels[:len(self.data)]

        # Take the second half of the train file as validation data
        elif split == 'val':
            self.data = self.data[int(train_proportion*len(self.data)):]
            self.patch_shape = -1
            self.random_crop = False
            self.labels = labels[len(self.data):]

        elif split == 'test':
            self.patch_shape = -1
            self.random_crop = False
            self.labels = labels
            # self.label_interpretation = pd.read_csv(self._interpretation_path, compression='gzip')

        if self.selected_features[0] == -1:
            if self.normalization_kind is not None:
                self.scaler.fit(X=self.data)
            # shape (num_samples, num_features)
            self.data_selected_features = np.atleast_2d(self.data)

        elif np.all([(0 <= i < self.data.shape[1]) for i in self.selected_features]) == True:
            if self.normalization_kind is not None:
                self.scaler.fit(X=self.data[:, self.selected_features])
            # shape (num_samples, num_features)
            self.data_selected_features = np.atleast_2d(self.data[:, self.selected_features])

        else:
            raise ValueError(f"Invalid value for selected_features: {selected_features}. Admitted value are [-1, ...] or a list of positive integers ")
        self.num_features = self.data_selected_features.shape[1]

        if self.normalization_kind is not None:
            self.data_selected_features = self.scaler.transform(self.data_selected_features)
        self.data_selected_features = self.data_selected_features.transpose()

        if window_length == -1:
            self.window_length = self.data_selected_features.shape[1]
            self.num_windows = 1

        elif window_length > 0:
            self.window_length = window_length
            self.num_windows = int(self.data_selected_features.shape[1] / self.window_length)

        else:
            raise ValueError(f'Invalid value of arg window_length {window_length}. Select [-1] or a integer >0 ')

        self.data_selected_features = self.data_selected_features[:, :self.num_windows * self.window_length]
        self.labels = self.labels[:self.num_windows * self.window_length]
        # self.label_interpretation = self.label_interpretation.iloc[:self.num_windows * self.window_length]

        # Extract start and end indices of anomalies
        len_labels = len(self.labels)
        arr_cond_1 = np.arange(len_labels)[np.where(self.labels == 1)[0]]
        l = len(arr_cond_1)

        b_start = arr_cond_1[(np.arange(l) - 1) % l]
        self.label1_start_idx = arr_cond_1[np.where(np.abs(b_start - arr_cond_1) > 1)[0]]

        b_end = arr_cond_1[(np.arange(l) + 1) % l]
        self.label1_end_idx = arr_cond_1[np.where(np.abs(b_end - arr_cond_1) > 1)[0]]

        self.labels_windows = np.any(self.labels.reshape(-1, window_length), axis=1)

    def __getitem__(self, index):
        # __getitem__ returns a tuple, where first entry contains raw waveform in [-1, 1]
        # datapoint = super().__getitem__(index)[0].float()

        datapoint = self.data_selected_features[:, index * self.window_length: (index + 1) * self.window_length]

        # Normalize data to lie in [0, 1]
        # if self.normalize:
        #     datapoint = (datapoint + 1) / 2

        # Extract only first num_waveform_samples from waveform
        # if self.num_secs != -1:
        #     # Shape (channels, num_waveform_samples)
        #     datapoint = datapoint[:, : self.num_waveform_samples]
        #

        return datapoint

    def __len__(self):
        return self.num_windows

    # TODO: Delete __iter__ and __next__ methods, not needed for torch.utils.data.Dataset
    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):

        # Stop iteration if limit is reached
        if self.iter >= self.num_windows:
            raise StopIteration
        # Else increment and return old value
        self.iter += 1

        return self[self.iter - 1]

if __name__ == '__main__':
    ar_time_series = ARTimeSeries(
                 split='train',
                 window_length=100,
                 selected_features=[-1],
                 normalization_kind='plus1div2',
                 train_proportion=0.85,
                 )

    ar_time_series_test = ARTimeSeries(
                 split='test',
                 window_length=100,
                 selected_features=[-1],
                 normalization_kind='plus1div2',
                 train_proportion=0.85,
                 )

    ALPHA = 0.8

    for i in range(2):
        plt.figure()

        plt.plot(ar_time_series.data_selected_features[i, :], alpha=ALPHA, label='train')
        plt.plot(ar_time_series_test.data_selected_features[i, :], alpha=ALPHA, label='test')
        plt.plot(ar_time_series_test.labels[:, i], lw=2, c='black', alpha=ALPHA, label='label')
    plt.legend()

    plt.show()
