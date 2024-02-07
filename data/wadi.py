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
from evaluation_utils import get_events
from affiliation.generics import convert_vector_to_events
import datetime

CHECK_LOAD_PATH = '../../datasets/WADI'

FILENAME_PREPROCESSED_ATTACK = 'WADI_attackdata_preprocessed.csv.gz'
FILENAME_PREPROCESSED_NORMAL = 'WADI_14days_preprocessed.csv.gz'

# https://drive.google.com/open?id=1_LnLF3iDl6Rrupr0BsejXPiiQeocZaKV
FILENAME_ORIGINAL_NORMAL = 'WADI_14days_ts.csv'

# https://drive.google.com/open?id=1WtFJs4_m4hpeytUrrIO5-MOilShVsDll
FILENAME_ORIGINAL_ATTACK = 'WADI_attackdata_ts.csv'

labels = []


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

class WADI(Dataset):
    """WADI dataset.
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
        normalization_kind: 'str' = 'minmax',
        selected_features = (0,),
        stride=-1,
        skip_first_n_samples: int = 21600,
        downsampling_factor: int = 1,
        *args,
        **kwargs
    ):
        # Initialize
        super().__init__() #*args, **kwargs)

        # Set dataset properties
        self.name = f'swat'
        self._base_path = pathlib.Path(kwargs['root'])
        self.split = split
        self.selected_features = np.array(selected_features)
        self.normalization_kind = normalization_kind
        self.skip_first_n_samples = skip_first_n_samples
        self.downsampling_factor = downsampling_factor
        self.stride = stride
        self.window_length = window_length

        print('self.downsampling_factor', self.downsampling_factor)
        print('self.skip_first_n_samples', self.skip_first_n_samples)

        self.preprocessed_train = (self._base_path / FILENAME_PREPROCESSED_NORMAL).exists()
        self.preprocessed_test = (self._base_path / FILENAME_PREPROCESSED_ATTACK).exists()

        if split in ('train', 'val') and self.preprocessed_train:
            print('Loading preprocessed data... (train-val)')
            self._path_data = self._base_path / FILENAME_PREPROCESSED_NORMAL
            _df_data_and_labels = pd.read_csv(self._path_data)
            # with open(self._base_path / 'SWaT_Dataset_Normal_v1_preprocessed.pkl', 'rb') as f:
            #     _df_data_and_labels = pickle.load(f)

            self._timestamp = _df_data_and_labels['Timestamp']
            # NaN columns

            self.labels = _df_data_and_labels['Normal/Attack'].values.astype(np.float32)
            self.data = _df_data_and_labels.drop(['Timestamp', 'Normal/Attack'], axis=1)
            self.columns = self.data.columns
            self.data = data_stats = self.data.values.astype(np.float32)

        elif split == 'test' and self.preprocessed_train and self.preprocessed_test:
            print('Loading preprocessed data... (test)')
            self._path_data = self._base_path / FILENAME_PREPROCESSED_ATTACK
            _df_data_and_labels = pd.read_csv(self._path_data) #, nrows=10000)

            data_stats = pd.read_csv(self._base_path / FILENAME_PREPROCESSED_NORMAL)

            self._timestamp = _df_data_and_labels['Timestamp']
            self.labels = _df_data_and_labels['Normal/Attack'].values.astype(np.float32)
            self.data = _df_data_and_labels.drop(["Timestamp", "Normal/Attack"], axis=1)
            self.columns = self.data.columns

            self.data = self.data.values.astype(np.float32)

            data_stats.drop(["Timestamp", "Normal/Attack"], axis=1, inplace=True)
            data_stats = data_stats.values.astype(np.float32)

        else:
            # prepare path and load data
            if split in ('train', 'val'):
                print('\t\t Loading raw data... (train-val)')
                self._path_data = self._base_path / FILENAME_ORIGINAL_NORMAL
                _df_data_and_labels = pd.read_csv(self._path_data, index_col=0) #, nrows=10000)
                self.labels = np.zeros(len(_df_data_and_labels))

                # nan_columns = ['2_P_001_STATUS', '2_P_002_STATUS', '2_LS_001_AL', '2_LS_002_AL']
            elif split == 'test':
                print('\t\t Loading raw data... (test)')
                self._path_data = self._base_path / FILENAME_ORIGINAL_ATTACK
                _df_data_and_labels = pd.read_csv(self._path_data, index_col=0)
                self.labels = _df_data_and_labels['Normal/Attack'].values.astype(np.float32)
                _df_data_and_labels.drop(['Normal/Attack'], axis=1, inplace=True)
            else:
                raise ValueError(f'Invalid value of split {split}. Select one of ("train", "val", "test")')

            self._timestamp = _df_data_and_labels['Timestamp']
            print('split', split)
            print(_df_data_and_labels.columns)
            _df_data_and_labels.drop(['Time', 'Date', 'Timestamp'], axis=1, inplace=True)

            # Isolated NaN values filled with the column's mean
            _df_data_and_labels.fillna(_df_data_and_labels.mean(), inplace=True)

            # NaN columns filled with zero
            _df_data_and_labels.fillna(0, inplace=True)

            # trim column names
            self.data = _df_data_and_labels.rename(columns=lambda x: x.strip())
            self.columns = [x[46:] for x in self.data.columns]  # remove column name prefixes

            # self.data = _df_data_and_labels.drop(["Timestamp", "Normal/Attack"], axis=1)
            # self.columns = self.data.columns

            # prepare data and data_stats
            if split in ('train', 'val'):
                print('Initializing data and data_stats...')
                # for col in list(self.data):
                #     self.data[col] = self.data[col].apply(lambda x: str(x).replace(",", "."))

                self.data = data_stats = self.data.values.astype(np.float32)
                # self.df_data_and_labels = self.df_data_and_labels.apply(lambda x: str(x).replace(",", "."), ).astype(float)
                # Transform all columns into float64

            elif split == 'test':
                print('Initializing data and data_stats...')
                print('self.data.columns', self.data.columns)

                self.data = self.data.values.astype(np.float32)

                data_stats = pd.read_csv(self._base_path / FILENAME_ORIGINAL_NORMAL, index_col=0)#, nrows=10000)
                data_stats.drop(['Time', 'Date', 'Timestamp'], axis=1, inplace=True)
                data_stats.fillna(data_stats.mean(), inplace=True)
                data_stats.fillna(0, inplace=True)
                print(f'data_stats.columns', data_stats.columns)
                data_stats = data_stats.values.astype(np.float32)

            else:
                raise ValueError(f'Invalid value of split {split}. Select one of ("train", "val", "test")')

            self._save_preprocessed()

        self.num_tot_features = self.data.shape[1]

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
            if skip_first_n_samples > 0:
                self.data = self.data[skip_first_n_samples:]
                self.labels = self.labels[skip_first_n_samples:]

            self.data = self.data[:int(train_proportion*len(self.data))]
            self.labels = self.labels[:int(train_proportion*len(self.labels))]

            self.patch_shape = patch_shape
            self.random_crop = patch_shape != -1

        # Take the second half of the train file as validation data
        elif split == 'val':
            self.data = self.data[int(train_proportion * len(self.data)):]
            self.labels = self.labels[int(train_proportion * len(self.labels)):]
            self.patch_shape = -1
            self.random_crop = False

        elif split == 'test':
            # if skip_first_n_samples > 0:
            #     self.data = self.data[skip_first_n_samples:]
            #     self.labels = self.labels[skip_first_n_samples:]
            self.patch_shape = -1
            self.random_crop = False

        if self.selected_features[0] == -1:
            if self.normalization_kind is not None:
                self.scaler.fit(X=data_stats)
            # shape (num_samples, num_features)
            self.data_selected_features = np.atleast_2d(self.data)

        elif np.all([(0 <= i < self.data.shape[1]) for i in self.selected_features]) == True:
            if self.normalization_kind is not None:
                self.scaler.fit(X=data_stats[:, self.selected_features])
            # shape (num_samples, num_features)
            self.data_selected_features = np.atleast_2d(self.data[:, self.selected_features])

        else:
            raise ValueError(f"Invalid value for selected_features: {selected_features}. Admitted value are [-1, ...] or a list of positive integers ")
        self.num_features = self.data_selected_features.shape[1]

        if self.normalization_kind is not None:
            self.data_selected_features = self.scaler.transform(self.data_selected_features)

        # From (num_samples, num_features) to (num_features, num_samples)
        self.data_selected_features = self.data_selected_features.transpose()

        # Apply downsampling to data and labels
        if self.downsampling_factor > 1:
            self.downsample()

        if window_length == -1:
            self.window_length = self.data_selected_features.shape[1]
            self.num_windows = 1

        elif window_length > 0:
            self.window_length = window_length
            self.stride = window_length if stride == -1 else stride
            self.num_windows = int(self.data_selected_features.shape[1] / self.stride)

        else:
            raise ValueError(f'Invalid value of arg window_length {window_length}. Select [-1] or a integer >0 ')

        # Drop the last samples if the number of samples is not a multiple of window_length
        self.data_selected_features = self.data_selected_features[:, :self.num_windows * self.window_length]
        self.labels = self.labels[:self.num_windows * self.window_length]

        # Extract start and end indices of anomalies
        len_labels = len(self.labels)
        arr_cond_1 = np.arange(len_labels)[np.where(self.labels == 1)[0]]
        l = len(arr_cond_1)

        b_start = arr_cond_1[(np.arange(l) - 1) % l]
        self.label1_start_idx = arr_cond_1[np.where(np.abs(b_start - arr_cond_1) > 1)[0]]

        b_end = arr_cond_1[(np.arange(l) + 1) % l]
        self.label1_end_idx = arr_cond_1[np.where(np.abs(b_end - arr_cond_1) > 1)[0]]

        if window_length != -1:
            self.labels_windows = np.any(self.labels.reshape(-1, window_length), axis=1)
        else:
            self.labels_windows = None

        self.true_events_single_COM = get_events(self.labels)
        self.true_events_win_COM = get_events(np.any(self.labels.reshape(-1, self.window_length), axis=1))

        # Format of true events of f1AFF different from f1COM
        self.true_events_single_AFF = convert_vector_to_events(self.labels)
        self.true_events_win_AFF = convert_vector_to_events(np.any(self.labels.reshape(-1, self.window_length), axis=1))

        print('self.data_selected_features.shape = ', self.data_selected_features.shape)

    def __getitem__(self, index):

        datapoint = self.data_selected_features[:, index * self.stride: (index * self.stride) + self.window_length]

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

    def _save_preprocessed(self):
        # save preprocessed data
        print(f'Saving preprocessed data to {self._base_path}')
        print('self.data.shape = ', self.data.shape)
        print(f'len(self.columns) = {len(self.columns)}')

        df_to_save = pd.DataFrame(self.data, columns=self.columns)

        df_to_save['Timestamp'] = self._timestamp
        df_to_save['Normal/Attack'] = self.labels

        if self.split in ('train', 'val'):
            df_to_save.to_csv(self._base_path / FILENAME_PREPROCESSED_NORMAL,
                              compression='gzip',
                              index=False)
        elif self.split == 'test':
            df_to_save.to_csv(self._base_path / FILENAME_PREPROCESSED_ATTACK,
                              compression='gzip',
                              index=False)
        else:
            raise ValueError(f'Invalid value of split {self.split}. Select one of ("train", "val", "test")')

    # https://stackoverflow.com/questions/47715337/check-if-timestamp-column-is-in-date-range-from-another-dataframe
    @staticmethod
    def __table_attack(data):
        """
        Helper function to create table of attack labelss
        Args:
            data: original data with date and time columns converted to pandas.Timestamp

        Returns:
            table_attack: table with start and end timestamps of attacks
            data: original data with added label columns of attack
        """

        table_attack = pd.DataFrame(columns=['start', 'end'], data=
        [
            [pd.Timestamp('07:25:00.000 PM 10/09/2017'), pd.Timestamp('07:50:16.000 PM 10/09/2017')],
            [pd.Timestamp('10:24:10.000 AM 10/10/2017'), pd.Timestamp('10:34:00.000 AM 10/10/2017')],
            [pd.Timestamp('10:55:00.000 AM 10/10/2017'), pd.Timestamp('11:24:00.000 AM 10/10/2017')],
            # [pd.Timestamp('11:07:46.000 AM 10/10/2017'), pd.Timestamp('11:12:15.000 AM 10/10/2017')], # Only in Attack description
            [pd.Timestamp('11:30:40.000 AM 10/10/2017'), pd.Timestamp('11:44:50.000 AM 10/10/2017')],
            [pd.Timestamp('01:39:30.000 PM 10/10/2017'), pd.Timestamp('01:50:40.000 PM 10/10/2017')],
            [pd.Timestamp('02:48:17.000 PM 10/10/2017'), pd.Timestamp('03:00:32.000 PM 10/10/2017')],
            # [pd.Timestamp('02:48:17.000 PM 10/10/2017'), pd.Timestamp('02:59:55.000 PM 10/10/2017')],
            # [pd.Timestamp('02:53:44.000 PM 10/10/2017'), pd.Timestamp('03:00:32.000 PM 10/10/2017')], # Only in Attack description
            [pd.Timestamp('05:40:00.000 PM 10/10/2017'), pd.Timestamp('05:49:40.000 PM 10/10/2017')],
            [pd.Timestamp('10:55:00.000 AM 10/11/2017'), pd.Timestamp('10:56:27.000 AM 10/11/2017')],
            [pd.Timestamp('11:17:54.000 AM 10/11/2017'), pd.Timestamp('11:31:20.000 AM 10/11/2017')],
            [pd.Timestamp('11:36:31.000 AM 10/11/2017'), pd.Timestamp('11:47:00.000 AM 10/11/2017')],
            [pd.Timestamp('11:59:00.000 AM 10/11/2017'), pd.Timestamp('12:05:00.000 PM 10/11/2017')],
            [pd.Timestamp('12:07:30.000 PM 10/11/2017'), pd.Timestamp('12:10:52.000 PM 10/11/2017')],
            [pd.Timestamp('12:16:00.000 PM 10/11/2017'), pd.Timestamp('12:25:36.000 PM 10/11/2017')],
            [pd.Timestamp('03:26:30.000 PM 10/11/2017'), pd.Timestamp('03:37:00.000 PM 10/11/2017')],
         ]
         )
        table_attack.apply(pd.Timestamp)

        def check_attack(timestamp):
            for index, row in table_attack.iterrows():
                if row['start'] <= timestamp <= row['end']:
                    return True
            return False

        data['Normal/Attack'] = data.apply(check_attack, axis=1)
        data['Normal/Attack'] = data['Normal/Attack'].astype(int)

        return data

    def downsample(self):
        """
        # Code taken from here https://github.com/d-ailin/GDN
        path scripts/process_wadi.py
        Downsample data and labels by taking the median of the data and the maximum of the labels
        in a window of size downsampling_factor
        """

        col_num, orig_len = self.data_selected_features.shape

        down_time_len = orig_len // self.downsampling_factor
        print('down_time_len', down_time_len)
        print('orig_len', orig_len)
        print('self.downsampling_factor', self.downsampling_factor)
        print('self.labels.shape', self.labels.shape)
        print('self.data_selected_features.shape', self.data_selected_features.shape)

        self.data_selected_features = self.data_selected_features[:, :down_time_len * self.downsampling_factor]
        print('self.data_selected_features.shape', self.data_selected_features.shape)
        self.data_selected_features = self.data_selected_features[:, :down_time_len * self.downsampling_factor] \
            .reshape(col_num, -1, self.downsampling_factor)
        self.data_selected_features = np.median(self.data_selected_features, axis=2).reshape(col_num, -1)

        self.labels = self.labels[:down_time_len * self.downsampling_factor].reshape(-1, self.downsampling_factor)
        # if exist anomalies, then this sample is abnormal
        self.labels = np.round(np.max(self.labels, axis=1))
        print('-' * 50)
        print('self.labels.shape', self.labels.shape)
        print('self.data_selected_features.shape', self.data_selected_features.shape)
        print('-' * 50)


def random_crop1d(data, patch_shape: int):
    if not (0 < patch_shape <= data.shape[-1]):
        raise ValueError(f"Invalid shapes. patch_shape = {patch_shape}; data.shape {data.shape}")
    width_from = random.randint(0, data.shape[-1] - patch_shape)
    return data[
        ...,
        width_from : width_from + patch_shape,
    ]


def test_case(dataset, ix, len_selected_features, window_length):
    pprint.pprint(f"{ix}) columns(dataset) = {len(dataset)}")
    for i, d in enumerate(dataset):
        assert d.shape == (len_selected_features, window_length), f"assert case {ix}: d.shape = {d.shape}"
        pprint.pprint(i)
        pprint.pprint(d[0, :10])
        if i > 10:
            break


if __name__ == '__main__':

    idx_case = 0
    NUM_SELECTED_FEATURES = 127
    train_proportion = 0.85
    # TRAIN_LENGTH = int(train_proportion * 495000)
    # VAL_LENGTH = int((1 - train_proportion) * 495000)
    # TEST_LENGTH = 449919
    import time
    d_stats = {}
    d_dataset = {}
    window_length = 100
    selected_features = tuple(range(127)) #(0, 1)

    from sklearn.utils.validation import check_is_fitted

    for split in ['train', 'test']:
        # for selected_features in range(127):
        #     d_stats[split] = {}
        #     d_stats[split][selected_features] = {}
            print('-'*60)
            print('selected_features = ', selected_features)

            start = time.time()
            dataset = WADI(root=CHECK_LOAD_PATH, split=split, window_length=window_length, selected_features=selected_features,
                           train_proportion=train_proportion)
            end = time.time() - start
            print('-'*30)
            print(f"Time to load {split} dataset: {end:.2f} seconds")
            print('-'*30)
            print(f'is_fitted(dataset.scaler) = {check_is_fitted(dataset.scaler)}')
            print(f"dataset.data.shape = {dataset.data.shape}")
            print(f"dataset.data_selected_features.shape = {dataset.data_selected_features.shape}")
            print(f"dataset.labels.shape = {dataset.labels.shape}")
            print(f"dataset.num_windows = {dataset.num_windows}")
            print(f'np.isnan(dataset.data).sum().sum() = {np.isnan(dataset.data).sum().sum()}')
            print(f'np.isnan(dataset.data_selected_features).sum().sum() = {np.isnan(dataset.data_selected_features).sum().sum()}')
            print('-'*60)
            d_stats[split] = pd.DataFrame(dataset.data_selected_features.transpose()).describe()
