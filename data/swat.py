import random
import numpy as np
import pprint
from torch.utils.data import Dataset
import pathlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import pandas as pd
import datatable as dt
from evaluation_utils import get_events
from affiliation.generics import convert_vector_to_events

CHECK_LOAD_PATH = '../../datasets/SWAT'

FILENAME_PREPROCESSED_NORMAL = 'SWaT_Dataset_Normal_v1_preprocessed.csv.gz'
FILENAME_PREPROCESSED_ATTACK = 'SWaT_Dataset_Attack_v0_preprocessed.csv.gz'

FILENAME_ORIGINAL_NORMAL = 'SWaT_Dataset_Normal_v1.csv'
FILENAME_ORIGINAL_ATTACK = 'SWaT_Dataset_Attack_v0.csv'

class Plus1Div2Scaler(BaseEstimator, TransformerMixin):
    """
    A transformer class that scales the input data by adding 1 and dividing by 2,
    and provides an inverse transformation to restore the original data.

    """
    def __init__(self):
        """
        Initialize the Plus1Div2Scaler object.
        """
        pass


    def transform(self, X, y=None):
        """
        Apply the scaling transformation to the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to be transformed.

        y : array-like, shape (n_samples,), default=None
            The target labels. This parameter is ignored.

        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_features)
            The transformed data.
        """
        X = (X + 1) / 2
        return X

    def inverse_transform(self, X, y=None):
        """
        Apply the inverse scaling transformation to the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The transformed data to be inverted.

        y : array-like, shape (n_samples,), default=None
            The target labels. This parameter is ignored.

        Returns:
        --------
        X_restored : array-like, shape (n_samples, n_features)
            The original data before scaling.
        """
        X = X * 2 - 1
        return X


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

class SWAT(Dataset):
    """
    Dataset class for the SWAT dataset.

    Args:
        window_length (int): The length of the sliding window for creating data samples. Default is -1.
        patch_shape (int): legacy parameter for coinpp. Default is -1.
        train_proportion (float): The proportion of the training data. Default is 0.85.
        split (str): The split of the dataset to use. Must be one of 'train', 'val', or 'test'. Default is 'train'.
        normalization_kind (str): The kind of normalization to apply to the data. Must be one of 'minmax',
            'standard', 'plus1div2', or None. Default is 'minmax'.
        selected_features (tuple): A tuple of indices representing the selected features. Default is (0,).
        stride (int): The stride value for the sliding window. Default is -1.
        downsampling_factor (int): The factor for downsampling the data. Default is 1.
        median_downsampling (bool): Whether to apply median downsampling. Default is False.
        labels_windows_kind (str): The kind of labeling for windows. Must be one of 'any' or 'majority'.
            Default is 'any'.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Attributes:
        name (str): The name of the dataset.
        _base_path (Path): The base path of the dataset.
        split (str): The split of the dataset being used.
        selected_features (ndarray): An array of selected feature indices.
        normalization_kind (str): The kind of normalization being applied.
        preprocessed_train (bool): Whether the training data has been preprocessed.
        preprocessed_test (bool): Whether the test data has been preprocessed.
        downsampling_factor (int): The factor for downsampling the data.
        median_downsampling (bool): Whether to apply median downsampling.
        scaler (object): The scaler object for data normalization.
        data (ndarray): The data samples.
        labels (ndarray): The labels for the data samples.
        columns (Index): The column names of the data.
        num_tot_features (int): The total number of features in the data.
        data_selected_features (ndarray): The data samples with selected features.
        num_features (int): The number of selected features.
        window_length (int): The length of the sliding window.
        stride (int): The stride value for the sliding window.
        num_windows (int): The number of windows in the data.
        label1_start_idx (ndarray): The start indices of anomalies.
        label1_end_idx (ndarray): The end indices of anomalies.
        labels_windows (ndarray): The labels for windows.
        true_events_single_COM (ndarray): The true events for single data samples.
        true_events_win_COM (ndarray): The true events for windows (COM format).
        true_events_single_AFF (ndarray): The true events for single data samples (AFF format).
        true_events_win_AFF (ndarray): The true events for windows (AFF format).

    Methods:
        _save_preprocessed(): Saves the preprocessed data.
        __getitem__(index): Retrieves the item at the given index.
        __len__(): Returns the total number of samples.
        __iter__(): Initializes the iterator.
        __next__(): Returns the next item in the iterator.

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
        skip_first_n_samples: int = 0,
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
            _df_data_and_labels = dt.fread(self._base_path / FILENAME_PREPROCESSED_NORMAL).to_pandas()

            self._timestamp = _df_data_and_labels['Timestamp']
            self.data = _df_data_and_labels.drop(["Timestamp", "Normal/Attack"], axis=1)
            self.labels = _df_data_and_labels['Normal/Attack'].values.astype(np.float32)
            self.columns = self.data.columns
            self.data = data_stats = self.data.values.astype(np.float32)

        elif split == 'test' and self.preprocessed_train and self.preprocessed_test:
            print('Loading preprocessed data... (test)')
            _df_data_and_labels = dt.fread(self._base_path / FILENAME_PREPROCESSED_ATTACK).to_pandas()

            data_stats = dt.fread(self._base_path / FILENAME_PREPROCESSED_NORMAL).to_pandas()

            self._timestamp = _df_data_and_labels['Timestamp']
            self.data = _df_data_and_labels.drop(["Timestamp", "Normal/Attack"], axis=1)
            self.labels = _df_data_and_labels['Normal/Attack'].values.astype(np.float32)
            self.columns = self.data.columns
            self.data = self.data.values.astype(np.float32)

            data_stats.drop(["Timestamp", "Normal/Attack"], axis=1, inplace=True)

            for col in list(data_stats):
                data_stats[col] = data_stats[col].apply(lambda x: str(x).replace(",", "."))

            data_stats = data_stats.values.astype(np.float32)

        else:
            # prepare path and load data
            if split in ('train', 'val'):
                print('\t\t Loading raw data... (train-val)')
                self._path_data = self._base_path / FILENAME_ORIGINAL_NORMAL
                _df_data_and_labels = pd.read_csv(self._path_data)
                # Transform all columns into float64

            elif split == 'test':
                print('\t\t Loading raw data... (test)')
                self._path_data = self._base_path / FILENAME_ORIGINAL_ATTACK
                _df_data_and_labels = pd.read_csv(self._path_data, sep=';')

            else:
                raise ValueError(f'Invalid value of split {split}. Select one of ("train", "val", "test")')

            self._timestamp = _df_data_and_labels['Timestamp']
            self.labels = (_df_data_and_labels['Normal/Attack'] != 'Normal').values.astype(np.float32)
            self.data = _df_data_and_labels.drop(["Timestamp", "Normal/Attack"], axis=1)
            self.columns = self.data.columns

            # prepare data and data_stats
            if split in ('train', 'val'):
                for col in list(self.data):
                    self.data[col] = self.data[col].apply(lambda x: str(x).replace(",", "."))
                self.data = self.data.astype(float)
                self.data = data_stats = self.data.values.astype(np.float32)
                # Transform all columns into float64

            elif split == 'test':
                for col in list(self.data):
                    self.data[col] = self.data[col].apply(lambda x: str(x).replace(",", "."))
                self.data = self.data.values.astype(np.float32)

                data_stats = pd.read_csv(self._base_path / f'SWaT_Dataset_Normal_v1.csv')
                data_stats.drop(["Timestamp", "Normal/Attack"], axis=1, inplace=True)
                for col in list(data_stats):
                    data_stats[col] = data_stats[col].apply(lambda x: str(x).replace(",", "."))
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

    def _save_preprocessed(self):

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

    def __getitem__(self, index):
        """
            Retrieve the item at the given index.
            Args:
                index (int): The index of the item.

            Returns:
                tuple: A tuple containing the data sample and the corresponding label.

        """
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
    NUM_SELECTED_FEATURES = 51
    train_proportion = 0.85
    TRAIN_LENGTH = int(train_proportion * 495000)
    VAL_LENGTH = int((1 - train_proportion) * 495000)
    TEST_LENGTH = 449919

    split = 'train'
    window_length = 100
    selected_features = (0, 1)
    import time
    start = time.time()
    dataset = SWAT(root=CHECK_LOAD_PATH, split=split, window_length=window_length, selected_features=selected_features,
                   train_proportion=train_proportion)
    end = time.time() - start
    print(f"Time to load dataset: {end:.2f} seconds")
    for selected_features in [(-1,), (0, 4, 6, 8), (0,1), (1,2)]:
        pprint.pprint(f"selected_features = {selected_features}")

        if selected_features[0] == -1:
            len_selected_features = NUM_SELECTED_FEATURES
        elif np.all([(0 <= i < NUM_SELECTED_FEATURES) for i in selected_features]) == True:
            len_selected_features = len(selected_features)
        else:
            raise ValueError(
                f"Invalid value for selected_features: {selected_features}. Admitted value are [-1] or a list of positive integers ")
        for split in ['train', 'val', 'test']:
            for window_length in [-1, 10, 100]:
                dataset = SWAT(root=CHECK_LOAD_PATH, split=split, window_length=window_length,selected_features=selected_features,
                               train_proportion=train_proportion)

                idx_case += 1
                if window_length == -1:
                    if split == 'train':
                        window_length = TRAIN_LENGTH
                    elif split == 'val':
                        window_length = VAL_LENGTH
                    else:
                        window_length = TEST_LENGTH
                print(f'window_length = {window_length}')
                print(f'len(dataset) = {len(dataset)}')
                print(f'split = {split}')
                print(f'selected_features = {selected_features}')
                print(dataset.__dict__)

                test_case(dataset, idx_case, len_selected_features, window_length=window_length)

    # # Check with window_length = -1
    # smd_dataset_train = SMD(root=CHECK_LOAD_PATH, patch_shape=200, entity=entity, split='train', window_length=-1,
    #                         selected_features=selected_features)
    # for i, d in enumerate(smd_dataset_train):
    #     assert d.shape == (len_selected_features, 200), f"assert case 6: d.shape = {d.shape}"
    #     pprint.pprint(i)
    #     pprint.pprint(d[0, :10])
    #     if i > 10:
    #         break
    swat_dataset_test = SWAT(root=CHECK_LOAD_PATH, patch_shape=-1, split='test', window_length=10, selected_features=(0, 1))

    global_index = 0

    y = swat_dataset_test.data_selected_features[global_index, :]
    labels = swat_dataset_test.labels

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
