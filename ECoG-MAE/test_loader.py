# TODO check filtering, envelope and resampling with Arnab, implement code such that we can flexibly load data from different patients

import numpy as np
import pandas as pd
import time as t
import os
import mne
from mne_bids import BIDSPath
import pyedflib
from pyedflib import highlevel
import scipy.signal
from einops import rearrange
import torch


class CustomImageDataset(torch.utils.data.Dataset):

    def __init__(self, args, root, path, bands, fs, new_fs):
        self.args = args
        self.root = root
        self.path = path
        self.bands = bands
        self.fs = fs
        self.new_fs = new_fs
        # since we take 2 sec samples, the number of samples we can stream from our dataset is determined by the duration of the chunk in sec divided by 2
        self.num_samples = highlevel.read_edf_header(edf_file=self.path)["Duration"] / 2
        if args.norm == "hour":
            self.means, self.stds = get_signal_stats(self.path)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        # here we define the grid - since for patient 798 grid electrodes are G1 - G64
        grid = np.linspace(1, 64, 64).astype(int)

        n_samples = int(2 * self.fs)
        n_new_samples = int(2 * self.new_fs)

        # load edf and extract signal
        raw = read_raw(self.path)
        sig = raw.get_data(
            picks=grid,
            start=(n_samples * self.index),
            stop=(n_samples * (self.index + 1)),
        )

        # zero pad if chunk is shorter than 2 sec
        if len(sig[0]) < n_samples:
            padding = np.zeros((64, n_samples - len(sig[0])))
            sig = np.concatenate((sig, padding), axis=1)

        # zero pad if channel is not included in grid #TODO a bit clunky right now, implement in a better and more flexible way
        # since we will load by index position of channel (so if a channel is not included it will load channel n+1 at position 1),
        # we correct that by inserting 0 at position n and shift value one upwards
        for i in range(0, 64):
            chn = "G" + str(i + 1)

            # first we check whether the channel is included
            if np.isin(chn, raw.info.ch_names) == False:
                # if not we insert 0 padding and shift upwards
                sig = np.insert(sig, i, np.zeros((1, n_samples)), axis=0)

        # delete items that were shifted upwards
        sig = sig[:64, :]

        # normalize signal within each 2 sec chunk - #TODO implement in vectorized form
        if self.args.norm == "sample":
            for ch in range(0, len(sig)):
                if np.std(sig[ch]) == 0:
                    continue
                else:
                    sig[ch] = sig[ch] - np.mean(sig[ch]) / np.std(sig[ch])
        elif self.args.norm == "hour":
            # TODO
            for ch in range(0, len(sig)):
                if np.std(sig[ch]) == 0:
                    continue
                else:
                    sig[ch] = sig[ch] - self.means[ch] / self.stds[ch]

        # extract frequency bands
        nyq = 0.5 * self.fs
        filtered = []

        for i in range(0, len(self.bands)):
            lowcut = self.bands[i][0]
            highcut = self.bands[i][1]
            # divide by self.fs instead? #TODO
            low = lowcut / nyq
            high = highcut / nyq

            sos = scipy.signal.butter(N=4, Wn=[low, high], btype="band", output="sos")
            filtered.append(scipy.signal.sosfilt(sos, sig))

        filtered = np.array(filtered)

        if self.args.env:
            # compute power envelope
            envelope = np.abs(scipy.signal.hilbert(filtered, axis=2))

            # look at power spectrum instead #TODOs

            # # decimate before - low pass filter if new_fs == 20 then < 10 Hz
            # sos = scipy.signal.butter(
            #     N=4, Wn=[0.5 * self.new_fs / 0.5 * self.new_fs], btype="low", output="sos"
            # )
            # envelope = scipy.signal.sosfilt(sos, envelope)

            # resample
            resampled = scipy.signal.resample(envelope, n_new_samples, axis=2)

        else:
            resampled = scipy.signal.resample(filtered, n_new_samples, axis=2)

        # try smoothing window averaging instead #TODO

        # rearrange into shape c*t*d*h*w, where
        # c = freq bands,
        # t = number of datapoints within a sample
        # d = depth (currently 1)
        # h = height of grid (currently 8)
        # w = width of grid (currently 8)
        out = rearrange(
            np.array(resampled, dtype=np.float32), "c (h w) t -> c t () h w", h=8, w=8
        )

        end = t.time()

        # print('Time elapsed: ' + str(end-start))

        return out


class ECoGDataset(torch.utils.data.IterableDataset):

    def __init__(self, args, root, path, bands, fs, new_fs):
        self.args = args
        self.root = root
        self.path = path
        self.bands = bands
        self.fs = fs
        self.new_fs = new_fs
        # since we take 2 sec samples, the number of samples we can stream from our dataset is determined by the duration of the chunk in sec divided by 2
        self.max_samples = highlevel.read_edf_header(edf_file=self.path)["Duration"] / 2
        if args.norm == "hour":
            self.means, self.stds = get_signal_stats(self.path)
        self.index = 0

    def __iter__(self):
        # this is to make sure we stop streaming from our dataset after the max number of samples is reached
        while self.index < self.max_samples:
            yield self.sample_data()
            self.index += 1
        # this is to reset the counter after we looped through the dataset so that streaming starts at 0 in the next epoch, since the dataset is not initialized again
        if self.index == self.max_samples:
            self.index = 0

    def sample_data(self):

        start = t.time()

        # here we define the grid - since for patient 798 grid electrodes are G1 - G64
        grid = np.linspace(1, 64, 64).astype(int)

        n_samples = int(2 * self.fs)
        n_new_samples = int(2 * self.new_fs)

        # load edf and extract signal
        raw = read_raw(self.path)
        sig = raw.get_data(
            picks=grid,
            start=(n_samples * self.index),
            stop=(n_samples * (self.index + 1)),
        )

        # zero pad if chunk is shorter than 2 sec
        if len(sig[0]) < n_samples:
            padding = np.zeros((64, n_samples - len(sig[0])))
            sig = np.concatenate((sig, padding), axis=1)

        # zero pad if channel is not included in grid #TODO a bit clunky right now, implement in a better and more flexible way
        # since we will load by index position of channel (so if a channel is not included it will load channel n+1 at position 1),
        # we correct that by inserting 0 at position n and shift value one upwards
        for i in range(0, 64):
            chn = "G" + str(i + 1)

            # first we check whether the channel is included
            if np.isin(chn, raw.info.ch_names) == False:
                # if not we insert 0 padding and shift upwards
                sig = np.insert(sig, i, np.zeros((1, n_samples)), axis=0)

        # delete items that were shifted upwards
        sig = sig[:64, :]

        # normalize signal within each 2 sec chunk - #TODO implement in vectorized form
        if self.args.norm == "sample":
            for ch in range(0, len(sig)):
                if np.std(sig[ch]) == 0:
                    continue
                else:
                    sig[ch] = sig[ch] - np.mean(sig[ch]) / np.std(sig[ch])
        elif self.args.norm == "hour":
            # TODO
            for ch in range(0, len(sig)):
                if np.std(sig[ch]) == 0:
                    continue
                else:
                    sig[ch] = sig[ch] - self.means[ch] / self.stds[ch]

        # extract frequency bands
        nyq = 0.5 * self.fs
        filtered = []

        for i in range(0, len(self.bands)):
            lowcut = self.bands[i][0]
            highcut = self.bands[i][1]
            # divide by self.fs instead? #TODO
            low = lowcut / nyq
            high = highcut / nyq

            sos = scipy.signal.butter(N=4, Wn=[low, high], btype="band", output="sos")
            filtered.append(scipy.signal.sosfilt(sos, sig))

        filtered = np.array(filtered)

        if self.args.env:
            # compute power envelope
            envelope = np.abs(scipy.signal.hilbert(filtered, axis=2))

            # look at power spectrum instead #TODOs

            # # decimate before - low pass filter if new_fs == 20 then < 10 Hz
            # sos = scipy.signal.butter(
            #     N=4, Wn=[0.5 * self.new_fs / 0.5 * self.new_fs], btype="low", output="sos"
            # )
            # envelope = scipy.signal.sosfilt(sos, envelope)

            # resample
            resampled = scipy.signal.resample(envelope, n_new_samples, axis=2)

        else:
            resampled = scipy.signal.resample(filtered, n_new_samples, axis=2)

        # try smoothing window averaging instead #TODO

        # rearrange into shape c*t*d*h*w, where
        # c = freq bands,
        # t = number of datapoints within a sample
        # d = depth (currently 1)
        # h = height of grid (currently 8)
        # w = width of grid (currently 8)
        out = rearrange(
            np.array(resampled, dtype=np.float32), "c (h w) t -> c t () h w", h=8, w=8
        )

        end = t.time()

        # print('Time elapsed: ' + str(end-start))

        return out


def get_signal_stats(path):

    reader = pyedflib.EdfReader(path)

    means = []
    stds = []

    for i in range(0, 64):

        if i in [58, 59, 60]:
            means.append(0)
            stds.append(0)
        else:
            signal = reader.readSignal(i, 0)
            means.append(np.mean(signal))
            stds.append(np.std(signal))

    return means, stds


def split_dataframe(args, df, ratio):
    """
    Shuffles a pandas dataframe and splits it into two dataframes with the specified ratio

    Args:
        df: The dataframe to split
        ratio: The proportion of data for the first dataframe (default: 0.9)

    Returns:
        df1: train split dataframe containing a proportion of ratio of full dataframe
        df2: test split dataframe containing a proportion of 1-ratio of the full dataframe
    """

    # # Shuffle the dataframe
    if args.shuffle:
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


def dl_setup(args):
    """
    Sets up dataloaders for train and test split. Here, we use a chain dataset implementation, meaning we concatenate 1 hour chunks of our data as iterable datasets into a larger
    dataset from which we can stream - https://discuss.pytorch.org/t/using-chaindataset-to-combine-iterabledataset/85236

    Args:
        args: command line arguments

    Returns:
        train_dl: dataloader instance for train split
        test_dl: dataloader instance for test split
    """

    dataset_path = os.path.join(os.getcwd(), args.dataset_path)
    root = os.path.join(dataset_path, "derivatives/preprocessed")
    data = pd.read_csv(os.path.join(dataset_path, "dataset.csv"))

    # only look at subset of data
    data = data.iloc[: int(len(data) * args.data_size), :]
    # data = data.iloc[int(len(data) * (1 - args.data_size)) :, :]
    train_data, test_data = split_dataframe(args, data, args.train_data_proportion)

    bands = args.bands
    fs = 512
    new_fs = args.new_fs
    batch_size = args.batch_size

    # load and concatenate data for train split
    train_datasets = []

    num_train_samples = 0

    trains = []

    for i, row in train_data.iterrows():
        path = BIDSPath(
            root=root,
            datatype="car",
            subject=f"{row.subject:02d}",
            task=f"part{row.task:03d}chunk{row.chunk:02d}",
            suffix="desc-preproc_ieeg",
            extension=".edf",
            check=False,
        )

        train_path = str(path.fpath)
        trains.append(
            {
                "name": train_path,
                "num_samples": int(
                    highlevel.read_edf_header(edf_file=train_path)["Duration"] / 2
                ),
            }
        )

        num_train_samples = num_train_samples + int(
            highlevel.read_edf_header(edf_file=train_path)["Duration"] / 2
        )

        train_datasets.append(ECoGDataset(args, root, train_path, bands, fs, new_fs))

    train_dataset_combined = torch.utils.data.ChainDataset(train_datasets)
    train_dl = torch.utils.data.DataLoader(
        train_dataset_combined, batch_size=batch_size
    )

    train_samples = pd.DataFrame(trains)

    # load and concatenate data for test split
    test_datasets = []

    tests = []

    for i, row in test_data.iterrows():
        path = BIDSPath(
            root=root,
            datatype="car",
            subject=f"{row.subject:02d}",
            task=f"part{row.task:03d}chunk{row.chunk:02d}",
            suffix="desc-preproc_ieeg",
            extension=".edf",
            check=False,
        )

        test_path = str(path.fpath)

        tests.append(
            {
                "name": test_path,
                "num_samples": int(
                    highlevel.read_edf_header(edf_file=test_path)["Duration"] / 2
                ),
            }
        )

        test_datasets.append(ECoGDataset(args, root, test_path, bands, fs, new_fs))

    test_dataset_combined = torch.utils.data.ChainDataset(test_datasets)
    test_dl = torch.utils.data.DataLoader(test_dataset_combined, batch_size=batch_size)

    test_samples = pd.DataFrame(tests)

    dir = os.getcwd() + f"/results/samples/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    train_samples.to_csv(
        dir + f"{args.job_name}_train_samples.csv",
        index=False,
    )

    test_samples.to_csv(
        dir + f"{args.job_name}_test_samples.csv",
        index=False,
    )

    return train_dl, test_dl, num_train_samples