import os
import pytest

import numpy as np
import pandas as pd
import scipy

from downstream_tasks.encoding_decoding.config import EncodingDecodingDataConfig
from downstream_tasks.encoding_decoding.load_signal import EncodingDecodingDataset

ELECTRODE_FILE_FORMAT_STR = "electrode_{elec_id}.mat"


@pytest.fixture
def create_fake_electrode_files_fn(tmp_path):
    def create_fake_electrode_files(signals: np.array):
        """Writes fake .mat files with given signals.

        Args:
            signals (np.array): Shape [num_electrodes, num_signals]. Writes one file for every electrode.
        """
        for i, signal in enumerate(signals):
            scipy.io.savemat(
                os.path.join(tmp_path, ELECTRODE_FILE_FORMAT_STR.format(elec_id=i + 1)),
                {"p1st": signal},
            )

        return tmp_path

    return create_fake_electrode_files


@pytest.fixture
def create_word_embedding_csv_fn(tmp_path):
    def create_word_embedding_csv(embeddings=np.zeros((5, 64)), onsets=np.zeros(5)):
        # Real data gives embeddings as strings so do that here.
        embeddings_strs = []
        for embedding in embeddings:
            embeddings_strs.append(np.array2string(embedding, separator=", "))
        dataframe = pd.DataFrame(data={"embeddings": embeddings_strs, "onset": onsets})
        path = os.path.join(tmp_path, "embeddings.csv")
        dataframe.to_csv(path)
        print(pd.read_csv(path))
        return path

    return create_word_embedding_csv


def test_correctly_locates_electrodes(
    create_fake_electrode_files_fn, create_word_embedding_csv_fn
):
    # Real data has a 1 dimension at end so include here as well.
    fake_signal = np.ones((64, 1000, 1))
    dataset_path = create_fake_electrode_files_fn(fake_signal)

    config = EncodingDecodingDataConfig(
        encoding_neural_data_folder=dataset_path,
        electrode_glob_path=ELECTRODE_FILE_FORMAT_STR,
        conversation_data_df_path=create_word_embedding_csv_fn(),
    )

    data_loader = EncodingDecodingDataset(config)

    actual_electrode_data = data_loader._load_grid_data()

    assert np.all(actual_electrode_data == np.ones((64, 1000)))


def test_correctly_pads_missing_electrodes(
    create_fake_electrode_files_fn, create_word_embedding_csv_fn
):
    # Real data has a 1 dimension at end so include here as well.
    fake_signal = np.ones((62, 1000, 1))
    dataset_path = create_fake_electrode_files_fn(fake_signal)

    config = EncodingDecodingDataConfig(
        encoding_neural_data_folder=dataset_path,
        electrode_glob_path=ELECTRODE_FILE_FORMAT_STR,
        conversation_data_df_path=create_word_embedding_csv_fn(),
    )

    data_loader = EncodingDecodingDataset(config)

    actual_electrode_data = data_loader._load_grid_data()

    assert np.all(actual_electrode_data[:62] == np.ones((62, 1000)))
    assert np.all(actual_electrode_data[62:] == np.zeros((2, 1000)))


def test_correctly_loads_word_and_neural_data(
    create_fake_electrode_files_fn, create_word_embedding_csv_fn
):
    # Set up fake signal so there are 2 seconds worth of positive integers for every word onset.
    # Onsets are stored in ms.
    onsets = np.array(range(5000, 30000, 5000))
    embeddings = np.random.rand(onsets.shape[0], 64)

    sampling_frequency = 512
    sample_duration = 2
    fake_signal = np.zeros((64, 1000 * sampling_frequency, 1))

    # Setup signal for every word to be the onset times the electrode index + 1
    for onset in onsets:
        word_onset_idx = sampling_frequency * onset // 1000
        end_sample_idx = (
            onset // 1000 * sampling_frequency + sample_duration * sampling_frequency
        )
        for i in range(64):
            fake_signal[i, word_onset_idx:end_sample_idx, :] = onset * (i + 1)

    word_data_path = create_word_embedding_csv_fn(embeddings=embeddings, onsets=onsets)
    dataset_path = create_fake_electrode_files_fn(fake_signal)

    config = EncodingDecodingDataConfig(
        encoding_neural_data_folder=dataset_path,
        electrode_glob_path=ELECTRODE_FILE_FORMAT_STR,
        conversation_data_df_path=word_data_path,
        sample_length=sample_duration,
        original_fs=sampling_frequency,
    )

    data_loader = EncodingDecodingDataset(config)

    num_examples = 0
    for i, (word_embedding, neural_data) in enumerate(data_loader):
        num_examples += 1

        assert np.allclose(word_embedding, embeddings[i])

        # Expected data must be reshaped to match VideoMAE model input shape.
        reshaped_expected_electrode_data = np.zeros(
            (1, sample_duration * config.new_fs, 1, 8, 8)
        )
        for j in range(64):
            electrode_row_idx = j // 8
            electrode_col_idx = j % 8
            reshaped_expected_electrode_data[
                0, :, 0, electrode_row_idx, electrode_col_idx
            ] = (np.ones(sample_duration * config.new_fs) * onsets[i] * (j + 1))

        assert np.allclose(neural_data, reshaped_expected_electrode_data)

    assert num_examples == onsets.shape[0]


def test_handles_positive_lag_correctly(
    create_fake_electrode_files_fn, create_word_embedding_csv_fn
):
    # Set up fake signal so there are 2 seconds worth of positive integers for after a second for every word onset.
    # Onsets are stored in ms.
    onsets = np.array(range(5000, 30000, 5000))
    embeddings = np.random.rand(onsets.shape[0], 64)

    sampling_frequency = 512
    sample_duration = 2
    fake_signal = np.zeros((64, 1000 * sampling_frequency, 1))

    # Setup signal for every word to be the onset times the electrode index + 1
    for onset in onsets:
        word_onset_idx = sampling_frequency * onset // 1000 + 1 * sampling_frequency
        end_sample_idx = (
            onset // 1000 * sampling_frequency
            + sample_duration * sampling_frequency
            + 1 * sampling_frequency
        )
        for i in range(64):
            fake_signal[i, word_onset_idx:end_sample_idx, :] = onset * (i + 1)

    word_data_path = create_word_embedding_csv_fn(embeddings=embeddings, onsets=onsets)
    dataset_path = create_fake_electrode_files_fn(fake_signal)

    config = EncodingDecodingDataConfig(
        encoding_neural_data_folder=dataset_path,
        electrode_glob_path=ELECTRODE_FILE_FORMAT_STR,
        conversation_data_df_path=word_data_path,
        sample_length=sample_duration,
        original_fs=sampling_frequency,
        lag=1000,
    )

    data_loader = EncodingDecodingDataset(config)

    num_examples = 0
    for i, (word_embedding, neural_data) in enumerate(data_loader):
        num_examples += 1

        assert np.allclose(word_embedding, embeddings[i])

        # Expected data must be reshaped to match VideoMAE model input shape.
        reshaped_expected_electrode_data = np.zeros(
            (1, sample_duration * config.new_fs, 1, 8, 8)
        )
        for j in range(64):
            electrode_row_idx = j // 8
            electrode_col_idx = j % 8
            reshaped_expected_electrode_data[
                0, :, 0, electrode_row_idx, electrode_col_idx
            ] = (np.ones(sample_duration * config.new_fs) * onsets[i] * (j + 1))

        assert np.allclose(neural_data, reshaped_expected_electrode_data)

    assert num_examples == onsets.shape[0]


def test_handles_negative_lag_correctly(
    create_fake_electrode_files_fn, create_word_embedding_csv_fn
):
    # Set up fake signal so there are 2 seconds worth of positive integers for a second beforeevery word onset.
    # Onsets are stored in ms.
    onsets = np.array(range(5000, 30000, 5000))
    embeddings = np.random.rand(onsets.shape[0], 64)

    sampling_frequency = 512
    sample_duration = 2
    fake_signal = np.zeros((64, 1000 * sampling_frequency, 1))

    # Setup signal for every word to be the onset times the electrode index + 1
    for onset in onsets:
        word_onset_idx = sampling_frequency * onset // 1000 - 1 * sampling_frequency
        end_sample_idx = (
            onset // 1000 * sampling_frequency
            + sample_duration * sampling_frequency
            - 1 * sampling_frequency
        )
        for i in range(64):
            fake_signal[i, word_onset_idx:end_sample_idx, :] = onset * (i + 1)

    word_data_path = create_word_embedding_csv_fn(embeddings=embeddings, onsets=onsets)
    dataset_path = create_fake_electrode_files_fn(fake_signal)

    config = EncodingDecodingDataConfig(
        encoding_neural_data_folder=dataset_path,
        electrode_glob_path=ELECTRODE_FILE_FORMAT_STR,
        conversation_data_df_path=word_data_path,
        sample_length=sample_duration,
        original_fs=sampling_frequency,
        lag=-1000,
    )

    data_loader = EncodingDecodingDataset(config)

    num_examples = 0
    for i, (word_embedding, neural_data) in enumerate(data_loader):
        num_examples += 1

        assert np.allclose(word_embedding, embeddings[i])

        # Expected data must be reshaped to match VideoMAE model input shape.
        reshaped_expected_electrode_data = np.zeros(
            (1, sample_duration * config.new_fs, 1, 8, 8)
        )
        for j in range(64):
            electrode_row_idx = j // 8
            electrode_col_idx = j % 8
            reshaped_expected_electrode_data[
                0, :, 0, electrode_row_idx, electrode_col_idx
            ] = (np.ones(sample_duration * config.new_fs) * onsets[i] * (j + 1))

        assert np.allclose(neural_data, reshaped_expected_electrode_data)

    assert num_examples == onsets.shape[0]


def test_can_handle_padding_signal_when_lag_is_before_signal_start(
    create_fake_electrode_files_fn, create_word_embedding_csv_fn
):
    # Set up fake signal so there is a signal at the start but lags will make the first signal start at a (-value)
    onsets = np.array([1000])
    embeddings = np.random.rand(onsets.shape[0], 64)

    sampling_frequency = 512
    sample_duration = 2
    fake_signal = np.zeros((64, 1000 * sampling_frequency, 1))

    # Add a second of integers to start of signal
    for i in range(64):
        fake_signal[i, 0 : 1 * sampling_frequency, :] = i + 1

    word_data_path = create_word_embedding_csv_fn(embeddings=embeddings, onsets=onsets)
    dataset_path = create_fake_electrode_files_fn(fake_signal)

    config = EncodingDecodingDataConfig(
        encoding_neural_data_folder=dataset_path,
        electrode_glob_path=ELECTRODE_FILE_FORMAT_STR,
        conversation_data_df_path=word_data_path,
        sample_length=sample_duration,
        original_fs=sampling_frequency,
        lag=-2000,
    )

    data_loader = EncodingDecodingDataset(config)

    num_examples = 0
    for i, (word_embedding, neural_data) in enumerate(data_loader):
        num_examples += 1

        assert np.allclose(word_embedding, embeddings[i])

        # Expected data must be reshaped to match VideoMAE model input shape.
        reshaped_expected_electrode_data = np.zeros(
            (1, sample_duration * config.new_fs, 1, 8, 8)
        )
        for j in range(64):
            electrode_row_idx = j // 8
            electrode_col_idx = j % 8
            # Signal should be a second of padded 0's and a second of 1's
            expected_signal = np.concatenate(
                [
                    np.zeros(int(sample_duration / 2 * config.new_fs)),
                    np.ones(int(sample_duration / 2 * config.new_fs)),
                ]
            )
            reshaped_expected_electrode_data[
                0, :, 0, electrode_row_idx, electrode_col_idx
            ] = expected_signal * (j + 1)

        assert np.allclose(neural_data, reshaped_expected_electrode_data)

    assert num_examples == onsets.shape[0]


def test_can_handle_padding_signal_when_end_of_sample_is_after_signal_ends(
    create_fake_electrode_files_fn, create_word_embedding_csv_fn
):
    # Set up fake signal so there is a signal at the end but lags will make the neural data end after the signal ends.
    onsets = np.array([999 * 1000])
    embeddings = np.random.rand(onsets.shape[0], 64)

    sampling_frequency = 512
    sample_duration = 2
    fake_signal = np.zeros((64, 1000 * sampling_frequency, 1))

    # Add a second of integers to start of signal
    for i in range(64):
        fake_signal[i, 999 * sampling_frequency :, :] = i + 1

    word_data_path = create_word_embedding_csv_fn(embeddings=embeddings, onsets=onsets)
    dataset_path = create_fake_electrode_files_fn(fake_signal)

    config = EncodingDecodingDataConfig(
        encoding_neural_data_folder=dataset_path,
        electrode_glob_path=ELECTRODE_FILE_FORMAT_STR,
        conversation_data_df_path=word_data_path,
        sample_length=sample_duration,
        original_fs=sampling_frequency,
        lag=0,
    )

    data_loader = EncodingDecodingDataset(config)

    num_examples = 0
    for i, (word_embedding, neural_data) in enumerate(data_loader):
        num_examples += 1

        assert np.allclose(word_embedding, embeddings[i])

        # Expected data must be reshaped to match VideoMAE model input shape.
        reshaped_expected_electrode_data = np.zeros(
            (1, sample_duration * config.new_fs, 1, 8, 8)
        )
        for j in range(64):
            electrode_row_idx = j // 8
            electrode_col_idx = j % 8
            # Signal should be a second of padded 0's and a second of 1's
            expected_signal = np.concatenate(
                [
                    np.ones(int(sample_duration / 2 * config.new_fs)),
                    np.zeros(int(sample_duration / 2 * config.new_fs)),
                ]
            )
            reshaped_expected_electrode_data[
                0, :, 0, electrode_row_idx, electrode_col_idx
            ] = expected_signal * (j + 1)

        assert np.allclose(neural_data, reshaped_expected_electrode_data)

    assert num_examples == onsets.shape[0]
