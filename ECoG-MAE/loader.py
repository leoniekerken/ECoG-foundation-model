#TODO check filtering, envelope and resampling with Arnab, implement code such that we can flexibly load data from different patients

import numpy as np
import pandas as pd
import time as t
import mne
from mne_bids import BIDSPath
from pyedflib import highlevel
import scipy.signal
from einops import rearrange
import torch


class ECoGDataset(torch.utils.data.IterableDataset):

    def __init__(self, root, path, bands, fs, new_fs):
        self.root = root
        self.path = path
        self.bands = bands
        self.fs = fs
        self.new_fs = new_fs
        # since we take 2 sec samples, the number of samples we can stream from our dataset is determined by the duration of the chunk in sec divided by 2
        self.max_samples = highlevel.read_edf_header(edf_file=self.path)["Duration"] / 2
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

        # not all samples are 2 secs #TODO
        n_samples = int(2 * self.fs)
        n_new_samples = int(2 * self.new_fs)

        # load edf and extract signal
        raw = read_raw(self.path)
        sig = raw.get_data(
            picks=grid,
            start=(n_samples * self.index),
            stop=(n_samples * (self.index + 1)),
        )

        norm_sig = sig.copy()

        # normalize signal within each 2 sec chunk - #TODO implement in vectorized form
        for ch in range(0, len(sig)):
            norm_sig[ch] = sig[ch] - np.mean(sig[ch]) / np.std(sig[ch])

        # zero pad if chunk is shorter than 2 sec
        if len(norm_sig[0]) < n_samples:  
            padding = np.zeros((64, n_samples - len(norm_sig[0])))
            norm_sig = np.concatenate((norm_sig, padding), axis=1)

        # zero pad if channel is not included in grid #TODO a bit clunky right now, implement in a better and more flexible way
        # since we will load by index position of channel (so if a channel is not included it will load channel n+1 at position 1),
        # we correct that by inserting 0 at position n and shift value one upwards
        for i in range(0, 64):
            chn = "G" + str(i + 1)

            # first we check whether the channel is included
            if np.isin(chn, raw.info.ch_names) == False:
                # if not we insert 0 padding and shift upwards
                norm_sig = np.insert(norm_sig, i, np.zeros((1, n_samples)), axis=0)

        # delete items that were shifted upwards
        norm_sig = norm_sig[:64, :] 

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
            filtered.append(scipy.signal.sosfilt(sos, norm_sig))

        filtered = np.array(filtered)

        # compute power envelope
        envelope = np.abs(scipy.signal.hilbert(filtered, axis=2))

        # look at power spectrum instead #TODO

        # decimate before - low pass filter if new_fs == 20 then < 10 Hz
        # resample
        resampled = scipy.signal.resample(envelope, n_new_samples, axis=2)

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
    

def split_dataframe(df, ratio):

    """
    Shuffles a pandas dataframe and splits it into two dataframes with the specified ratio

    Args:
        df: The dataframe to split
        ratio: The proportion of data for the first dataframe (default: 0.9)
        
    Returns:
        df1: train split dataframe containing a proportion of ratio of full dataframe
        df2: test split dataframe containing a proportion of 1-ratio of the full dataframe
    """

    # Shuffle the dataframe
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
    
    #TODO change to point to sandbox data
    root = "/scratch/gpfs/ln1144/fm-preproc/dataset/derivatives/preprocessed"
    data = pd.read_csv("/scratch/gpfs/ln1144/fm-preproc/dataset/dataset.csv")

    # only look at subset of data
    data = data.iloc[: int(len(data) * args.data_size), :]

    train_data, test_data = split_dataframe(data, 0.9)

    bands = args.bands
    fs = 512
    new_fs = args.new_fs
    batch_size = args.batch_size

    # load and concatenate data for train split
    train_datasets = []

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

        train_datasets.append(ECoGDataset(root, train_path, bands, fs, new_fs))

    train_dataset_combined = torch.utils.data.ChainDataset(train_datasets)
    train_dl = torch.utils.data.DataLoader(
        train_dataset_combined, batch_size=batch_size
    )

    # load and concatenate data for test split
    test_datasets = []

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

        test_datasets.append(ECoGDataset(root, test_path, bands, fs, new_fs))

    test_dataset_combined = torch.utils.data.ChainDataset(test_datasets)
    test_dl = torch.utils.data.DataLoader(test_dataset_combined, batch_size=batch_size)

    return train_dl, test_dl