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

CHECK_LOAD_PATH = '../../datasets/smd'


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

class SMD(Dataset):
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
        entity: str = 'machine-1-1',
        split: str = 'train',
        normalization_kind: 'str' = 'plus1div2',
        selected_features = (0,),
        *args,
        **kwargs
    ):
        super().__init__() #*args, **kwargs)

        self.name = f'smd-{entity}'
        self._base_path = pathlib.Path(kwargs['root'])
        self.split = split
        self.selected_features = np.array(selected_features)

        if split in ('train', 'val'):
            self._path_data = self._base_path / f'{entity}_train.pkl'
        elif split == 'test':
            self._path_data = self._base_path / f'{entity}_test.pkl'
            self._path_labels = self._base_path / f'{entity}_test_label.pkl'
            self._interpretation_path = self._base_path / f'{entity}_test_interpretation.csv.gz'

        else:
            raise ValueError(f'Invalid value of split {split}. Select one of ("train", "val", "test")')

        self.normalization_kind = normalization_kind

        with self._path_data.open('rb') as fdata:
            self.data = pickle.load(fdata)
        self.num_tot_features = self.data.shape[1]

        if split not in ('train', 'val'):
            self._path_train_for_stats = self._base_path / f'{entity}_train.pkl'

            with self._path_train_for_stats.open('rb') as ftrain:
                data_stats = pickle.load(ftrain)
        else:
            data_stats = self.data

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
            self.labels = np.zeros(len(self.data))

        # Take the second half of the train file as validation data
        elif split == 'val':
            self.data = self.data[int(train_proportion*len(self.data)):]
            self.patch_shape = -1
            self.random_crop = False
            self.labels = np.zeros(len(self.data))

        elif split == 'test':
            self.patch_shape = -1
            self.random_crop = False
            with self._path_labels.open('rb') as flabels:
                self.labels = pickle.load(flabels)
            self.label_interpretation = pd.read_csv(self._interpretation_path, compression='gzip')

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

        if window_length != -1:
            self.labels_windows = np.any(self.labels.reshape(-1, window_length), axis=1)
        else:
            self.labels_windows = None

        self.true_events_single_COM = get_events(self.labels)
        self.true_events_win_COM = get_events(np.any(self.labels.reshape(-1, self.window_length), axis=1))

        # Format of true events of f1AFF different from f1COM
        self.true_events_single_AFF = convert_vector_to_events(self.labels)
        self.true_events_win_AFF = convert_vector_to_events(np.any(self.labels.reshape(-1, self.window_length), axis=1))

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
        if self.random_crop:
            datapoint = random_crop1d(datapoint, self.patch_shape)

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


def random_crop1d(data, patch_shape: int):
    if not (0 < patch_shape <= data.shape[-1]):
        raise ValueError(f"Invalid shapes. patch_shape = {patch_shape}; data.shape {data.shape}")
    width_from = random.randint(0, data.shape[-1] - patch_shape)
    return data[
        ...,
        width_from : width_from + patch_shape,
    ]


if __name__ == '__main__':

    window_length = 100
    smd_dataset_train = SMD(root=CHECK_LOAD_PATH, patch_shape=-1, split='train',
                            window_length=100)
    for i, d in enumerate(smd_dataset_train):
        assert d.shape == (1, window_length), f"assert case 0: d.shape = {d.shape}"
        pprint.pprint(i)
        pprint.pprint(d[0, :10])
        if i > 10:
            break
    1/0
    NUM_SELECTED_FEATURES = 38
    entity = 'machine-1-1'

    for i, d in enumerate(smd_dataset_train):
        assert d.shape == (1, 200), f"assert case 0: d.shape = {d.shape}"
        pprint.pprint(i)
        pprint.pprint(d[0, :10])
        if i > 10:
            break

    NUM_SELECTED_FEATURES = 38
    entity = 'machine-1-1'

    # for selected_features in [(-1,), (0, 4, 6, 8), (0,1), (1,2)]:
    #     pprint.pprint(f"selected_features = {selected_features}")
    #
    #     if selected_features[0] == -1:
    #         len_selected_features = NUM_SELECTED_FEATURES
    #     elif np.all([(0 <= i < NUM_SELECTED_FEATURES) for i in selected_features]) == True:
    #         len_selected_features = len(selected_features)
    #     else:
    #         raise ValueError(
    #             f"Invalid value for selected_features: {selected_features}. Admitted value are [-1] or a list of positive integers ")
    #
    #     # Check with window_length = -1
    #     smd_dataset_train = SMD(root=CHECK_LOAD_PATH, patch_shape=200, entity=entity, split='train', window_length=-1,
    #                             selected_features=selected_features)
    #     for i, d in enumerate(smd_dataset_train):
    #         assert d.shape == (len_selected_features, 200), f"assert case 1: d.shape = {d.shape}"
    #         pprint.pprint(i)
    #         pprint.pprint(d[0, :10])
    #         if i > 10:
    #             break
    #
    #     # Check with window_length = 1000
    #     smd_dataset_train = SMD(root=CHECK_LOAD_PATH, patch_shape=200, entity=entity, split='train', window_length=1000,
    #                             selected_features = selected_features)
    #     for i, d in enumerate(smd_dataset_train):
    #         assert d.shape == (len_selected_features, 200), f"assert case 2: d.shape = {d.shape}"
    #         pprint.pprint(i)
    #         pprint.pprint(d[0, :10])
    #         if i > 10:
    #             break
    #
    #     # Check with validation
    #     smd_dataset_train = SMD(root=CHECK_LOAD_PATH, patch_shape=200, entity=entity, split='val', window_length=1000,
    #                             selected_features = selected_features)
    #     for i, d in enumerate(smd_dataset_train):
    #         assert d.shape == (len_selected_features, 1000), f"assert case 3: d.shape = {d.shape}"
    #         pprint.pprint(i)
    #         pprint.pprint(d[0, :10])
    #         if i > 10:
    #             break
    #
    #     smd_dataset_test = SMD(root=CHECK_LOAD_PATH, patch_shape=200, entity=entity, split='test', window_length=-1,
    #                            selected_features=selected_features)
    #     for i, d in enumerate(smd_dataset_test):
    #         assert d.shape == (len_selected_features, 28479), f"assert case 4: d.shape = {d.shape}"
    #         pprint.pprint(i)
    #         pprint.pprint(d[0, :10])
    #         if i > 10:
    #             break
    #
    #     smd_dataset_test = SMD(root=CHECK_LOAD_PATH, patch_shape=200, entity=entity, split='test', window_length=1000,
    #                            selected_features = selected_features)
    #
    #     for i, d in enumerate(smd_dataset_test):
    #         assert d.shape == (len_selected_features, 1000), f"assert case 5: d.shape = {d.shape}"
    #         pprint.pprint(i)
    #         pprint.pprint(d[0, :10])
    #         if i > 10:
    #             break
    #
    # # Check with window_length = -1
    # smd_dataset_train = SMD(root=CHECK_LOAD_PATH, patch_shape=200, entity=entity, split='train', window_length=-1,
    #                         selected_features=selected_features)
    # for i, d in enumerate(smd_dataset_train):
    #     assert d.shape == (len_selected_features, 200), f"assert case 6: d.shape = {d.shape}"
    #     pprint.pprint(i)
    #     pprint.pprint(d[0, :10])
    #     if i > 10:
    #         break
    smd_dataset_test = SMD(root=CHECK_LOAD_PATH, patch_shape=-1, split='test', window_length=10, selected_features=(0, 1))

    global_index = 0

    y = smd_dataset_test.data_selected_features[global_index, :]
    labels = smd_dataset_test.labels

    arr_cond_0 = labels == 0
    arr_cond_1 = labels == 1

    y0 = np.ma.masked_where(arr_cond_1, y)
    y1 = np.ma.masked_where(arr_cond_0, y)

    t = np.arange(0, len(labels))

    # c = np.array(['#121212'] * len(labels))
    # c[arr_cond_1] = np.array('#ff0000')

    fig, ax = plt.subplots()

    ax.plot(t, y0, 'orange')
    ax.plot(t, y1, 'r')

    plt.ylabel('value')
    plt.xlabel('time')
    plt.show()
