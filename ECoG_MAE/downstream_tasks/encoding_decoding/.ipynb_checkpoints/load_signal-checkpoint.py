# Here we want to load the signal for the grid electrodes and organize it in an 8*8 matrix. We also want to perform filtering here - we might need to think about how to do it
# such that it is the same as the filtering we perform for model training - since there we do it using mne, which we can't here, since the preprocessed signal here are mat files.

import os
import glob
import numpy as np
import pandas as pd
import scipy

from utils import get_signal_stats, preprocess_neural_data
from downstream_tasks.encoding_decoding.config import EncodingDecodingDataConfig


class EncodingDecodingDataset:
    def __init__(self, config: EncodingDecodingDataConfig):
        self.config = config
        self.encoding_neural_data_folder = config.encoding_neural_data_folder
        self.electrode_glob_path = config.electrode_glob_path
        self.fs = config.original_fs
        self.lag = config.lag
        self.new_fs = config.new_fs
        self.sample_secs = config.sample_length

        self.signal = self._load_grid_data()
        conversation_data = pd.read_csv(config.conversation_data_df_path, index_col=0)
        # Convert embeddings from string to array. Index out "[]" from string.
        conversation_data.loc[:, "embeddings"] = conversation_data.loc[
            :, "embeddings"
        ].apply(lambda x: np.fromstring(x[1:-1], sep=", "))

        # Limit only to words with onset time data.
        self.timed_word_data = conversation_data.loc[
            conversation_data["onset"].dropna().index
        ]

        # since we take sample_length sec samples, the number of samples we can stream from our dataset is determined by the duration of the chunk in sec divided by sample_length.
        # Optionally can configure max_samples directly as well.
        self.max_samples = self.signal.shape[1] / self.fs / config.sample_length
        if config.norm == "hour":
            self.means, self.stds = get_signal_stats(self.signal)
        else:
            self.means = None
            self.stds = None

        self.index = 0

    def _load_grid_data(self):
        grid_data = []
        # Used to ensure all electrode data is of the same length, and to pad 0's later if needed.
        expected_len = 0
        for i in range(64):
            curr_electrode_glob_path = self.electrode_glob_path.format(
                elec_id=str(i + 1)
            )
            final_glob_path = os.path.join(
                self.encoding_neural_data_folder, curr_electrode_glob_path
            )
            electrode_file = glob.glob(final_glob_path)

            if len(electrode_file) > 1:
                raise ValueError(
                    "There can only be one matching file associated with electrode {}. Got {} files matching {}.".format(
                        i + 1, len(electrode_file), final_glob_path
                    )
                )
            elif len(electrode_file) == 0:
                print(
                    "No files found for electrode {}. Got 0 files matching {}.".format(
                        i + 1, final_glob_path
                    )
                )
                # Append None to be padded with 0's later.
                grid_data.append(None)
            else:
                data = scipy.io.loadmat(electrode_file[0])["p1st"].flatten()

                if not expected_len:
                    expected_len = data.size
                else:
                    if data.size != expected_len:
                        raise ValueError(
                            "Data size does not match for electrode {} at path: {}. Expected size: {}. Actual size: {}".format(
                                i + 1, final_glob_path, expected_len, data.size
                            )
                        )

                grid_data.append(data)
        padded_data = []
        for data in grid_data:
            # Pad zero's for held out data.
            if data is None:
                padded_data.append(np.zeros((expected_len)))
            else:
                padded_data.append(data)

        return np.array(padded_data)

    def __iter__(self):
        """Iterate through dataset for encoding task.

        Yields:
            tuple[np.array, np.array]: (word embedding, neural embedding)
        """
        while self.index < self.timed_word_data.shape[0]:
            word_data = self.timed_word_data.iloc[self.index]

            # TODO: Add configurable way to filter out examples which require padding if we want to train
            # without them. Same for VideoMAE dataloader.
            lag_start_time = word_data.loc["onset"] + self.lag
            lag_start_sample = int(lag_start_time / 1000 * self.fs)
            lag_end_sample = lag_start_sample + self.fs * self.sample_secs
            
            # If we are gathering a sample from before the start of the signal, pad before the sample.
            pad_before_sample = False
            if lag_start_sample < 0:
                pad_before_sample = True
                lag_start_sample = 0
                
            curr_sample = self.signal[
                :, lag_start_sample : lag_end_sample
            ]

            preprocessed_signal = preprocess_neural_data(
                curr_sample,
                self.fs,
                self.new_fs,
                self.sample_secs,
                pad_before_sample=pad_before_sample,
            )

            yield word_data.loc["embeddings"], preprocessed_signal

            self.index += 1

        if self.index >= self.timed_word_data.shape[0]:
            self.index = 0
