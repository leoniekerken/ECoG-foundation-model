# TODO check filtering, envelope and resampling with Arnab, implement code such that we can flexibly load data from different patients

import numpy as np
import pandas as pd
import time as t
import os
import mne
from mne_bids import BIDSPath
from pyedflib import highlevel
import torch
import logging

from config import ECoGDataConfig, VideoMAEExperimentConfig
from utils import preprocess_neural_data, get_signal_stats

logger = logging.getLogger(__name__)


class ECoGDataset(torch.utils.data.IterableDataset):

    def __init__(self, path: str, config: ECoGDataConfig):
        self.config = config
        self.path = path
        self.bands = config.bands
        self.fs = config.original_fs
        self.new_fs = config.new_fs
        self.sample_secs = config.sample_length

        # Temporarily load data to get stats. Don't maintain pointer though to avoid RAM overuse.
        signal = self._load_grid_data()
        # since we take sample_length sec samples, the number of samples we can stream from our dataset is determined by the duration of the chunk in sec divided by sample_length.
        # Optionally can configure max_samples directly as well.
        self.max_samples = signal.shape[1] / self.fs / config.sample_length
        if config.norm == "hour":
            self.means, self.stds = get_signal_stats(signal)
        else:
            self.means = None
            self.stds = None

        self.index = 0
    
    def __len__(self):
        return int(self.max_samples)

    def __iter__(self):
        # Load data into ram on first iteration.
        if self.index == 0:
            logger.debug("-----------------------------------------------------------------------------")
            logger.debug("Reading new file: %s", self.path)
            self.signal = self._load_grid_data()
        # this is to make sure we stop streaming from our dataset after the max number of samples is reached
        while self.index < self.max_samples:
            # Exclude examples where the sample goes past the end of the signal.
            start_sample = self.index * self.sample_secs * self.fs
            end_sample = (self.index + 1) * self.sample_secs * self.fs
            if end_sample > self.signal.shape[1]:
                self.index = self.max_samples
                break
                
            yield self.sample_data(start_sample, end_sample)
            self.index += 1
        # this is to reset the counter after we looped through the dataset so that streaming starts at 0 in the next epoch,
        # since the dataset is not initialized again. Also destroy pointer to data to free RAM.
        if self.index >= self.max_samples:
            self.index = 0
            del self.signal

    def sample_data(self, start_sample, end_sample) -> np.array:

        start = t.time()

        current_sample = self.signal[:, start_sample:end_sample]

        preprocessed_signal = preprocess_neural_data(
            current_sample,
            self.fs,
            self.new_fs,
            self.sample_secs,
            bands=self.bands,
            norm=self.config.norm,
            means=self.means,
            stds=self.stds,
        )

        end = t.time()

        # print('Time elapsed: ' + str(end-start))

        return preprocessed_signal

    def _load_grid_data(self):
        """Overridable function to load data from an mne file and return it in an unprocessed grid.

        Can be overridden to support different data types. Data will be preprocessed in the same way and returned via iteration over the dataset.

        Returns:
            numpy array of shape [number of electrodes, num_samples].
        """

        # load edf and extract signal
        raw = read_raw(self.path)

        # here we define the grid - since for patient 798 grid electrodes are G1 - G64
        grid_ch_names = []
        for i in range(64):
            channel = "G" + str(i + 1)
            if np.isin(channel, raw.info.ch_names):
                grid_ch_names.append(channel)

        sig = raw.get_data(
            picks=grid_ch_names,
        )
        n_samples = sig.shape[1]

        # zero pad if channel is not included in grid #TODO a bit clunky right now, implement in a better and more flexible way
        # since we will load by index position of channel (so if a channel is not included it will load channel n+1 at position 1),
        # we correct that by inserting 0 at position n and shift value one upwards
        for i in range(64):
            channel = "G" + str(i + 1)
            # first we check whether the channel is included
            if not np.isin(channel, raw.info.ch_names):
                # if not we insert 0 padding and shift upwards
                sig = np.insert(sig, i, np.zeros((n_samples), dtype=np.float32), axis=0)

        # delete items that were shifted upwards
        sig = sig[:64, :]
        
        # Make sure signal is float32
        sig = np.float32(sig)

        return sig


def split_dataframe(shuffle: bool, df: pd.DataFrame, ratio: float):
    """
    Shuffles a pandas dataframe and splits it into two dataframes with the specified ratio

    Args:
        shuffle: If true then shuffle the dataframe before splitting.
        df: The dataframe to split
        ratio: The proportion of data for the first dataframe (default: 0.9)

    Returns:
        df1: train split dataframe containing a proportion of ratio of full dataframe
        df2: test split dataframe containing a proportion of 1-ratio of the full dataframe
    """

    # # Shuffle the dataframe
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    # Calculate the split index based on the ratios
    split_index = int(ratio * len(df))

    # Create the two dataframes
    df1 = df.iloc[:split_index, :]
    df2 = df.iloc[split_index:, :]

    return df1, df2


def read_raw(filename):
    """
    Reads and loads an edf file into a mne raw object: https://mne.tools/stable/auto_tutorials/raw/10_raw_overview.html

    Args:
        filename: Path to edf file

    Returns:
        raw: a mne raw instance
    """

    raw = mne.io.read_raw(filename, verbose=False)

    return raw


def get_dataset_path_info(
    sample_length: int, root: str, data_split: pd.DataFrame
) -> tuple[list[str], int, pd.DataFrame]:
    """Generates information about the data referenced in data_split.

    Args:
        sample_length (int): number of seconds for each sample
        root (str): Filepath to root of BIDS dataset.
        data_split (pd.DataFrame): Dataframe storing references to the files to be used in this data split. Should have columns subject, task, and chunk.

    Returns: (List of filepaths to be used for data_split, Number of  samples for the data split, Dataframe with columns {'name': <filepath>, 'num_samples': <number of samples in file>})
    """
    split_filepaths = []

    num_samples = 0

    sample_desc = []

    for i, row in data_split.iterrows():
        path = BIDSPath(
            root=root,
            datatype="car",
            subject=f"{row.subject:02d}",
            task=f"part{row.task:03d}chunk{row.chunk:02d}",
            suffix="desc-preproc_ieeg",
            extension=".edf",
            check=False,
        )

        data_path = str(path.fpath)
        sample_desc.append(
            {
                "name": data_path,
                "num_samples": int(
                    highlevel.read_edf_header(edf_file=data_path)["Duration"] / sample_length
                ),
            }
        )

        num_samples = num_samples + int(
            highlevel.read_edf_header(edf_file=data_path)["Duration"] / sample_length
        )

        split_filepaths.append(data_path)

    return split_filepaths, num_samples, pd.DataFrame(sample_desc)


def _create_dataloader(root: str, data_files_df: pd.DataFrame, ecog_data_config: ECoGDataConfig) -> tuple[torch.utils.data.DataLoader, int, pd.DataFrame]:
    """Given a dataframe containing the BIDS data info in a dataset and the data config, create a dataloader and associated information.

    Args:
        data_files_df (pd.DataFrame): Has columns subject, task, and chunk for finding desired data in BIDS format.
        ecog_data_config (ECoGDataConfig): Configuration for how to preprocess data.

    Returns:
        tuple[torch.utils.data.DataLoader, int, pd.DataFrame]: [Dataloader for data, number of samples in dataloader, descriptions of how many samples are in each file]
    """
    # load and concatenate data for train split
    filepaths, num_samples, sample_desc = get_dataset_path_info(
        ecog_data_config.sample_length, root, data_files_df
    )
    datasets = [
        ECoGDataset(train_path, ecog_data_config)
        for train_path in filepaths
    ]
    dataset_combined = torch.utils.data.ChainDataset(datasets)
    dataloader = torch.utils.data.DataLoader(
        dataset_combined, batch_size=ecog_data_config.batch_size
    )
    
    return dataloader, num_samples, sample_desc
    

def dl_setup(
    config: VideoMAEExperimentConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    Sets up dataloaders for train and test split. Here, we use a chain dataset implementation, meaning we concatenate 1 hour chunks of our data as iterable datasets into a larger
    dataset from which we can stream - https://discuss.pytorch.org/t/using-chaindataset-to-combine-iterabledataset/85236

    Args:
        config: command line arguments

    Returns:
        train_dl: dataloader instance for train split
        test_dl: dataloader instance for test split
    """

    dataset_path = os.path.join(os.getcwd(), config.ecog_data_config.dataset_path)
    root = os.path.join(dataset_path, "derivatives/preprocessed")
    data = pd.read_csv(os.path.join(dataset_path, "dataset.csv"))

    # only look at subset of data
    data = data.iloc[: int(len(data) * config.ecog_data_config.data_size), :]
    # data = data.iloc[int(len(data) * (1 - config.ecog_data_config.data_size)) :, :]
    train_data, test_data = split_dataframe(
        config.ecog_data_config.shuffle,
        data,
        config.ecog_data_config.train_data_proportion,
    )
    
    train_dl, num_train_samples, train_samples_desc = _create_dataloader(root, train_data, config.ecog_data_config)
    test_dl, _, test_samples_desc = _create_dataloader(root, test_data, config.ecog_data_config)

    dir = os.getcwd() + f"/results/samples/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    train_samples_desc.to_csv(
        dir + f"{config.job_name}_train_samples.csv",
        index=False,
    )

    test_samples_desc.to_csv(
        dir + f"{config.job_name}_test_samples.csv",
        index=False,
    )

    return train_dl, test_dl, num_train_samples
