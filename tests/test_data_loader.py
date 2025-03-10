import numpy as np

from config import ECoGDataConfig


NUM_CHANNELS = 64
FILE_SAMPLING_FREQUENCY = 512


def create_fake_sin_data():
    # Create fake data to be used in tests. Sin wave scaled by index of channel
    # Add sine wave scaled by electrode index to all channels.
    times = np.linspace(0, 100, 100 * FILE_SAMPLING_FREQUENCY)
    data = np.array([(i + 1) * np.sin(np.pi * times) for i in range(NUM_CHANNELS + 1)])
    return data


def test_data_loader_can_handle_configurable_bands(data_loader_creation_fn):
    config = ECoGDataConfig(
        batch_size=32, bands=[[4, 8], [8, 13], [13, 30], [30, 55]], new_fs=20,
    )
    data_loader = data_loader_creation_fn(config, data=create_fake_sin_data(), file_sampling_frequency=FILE_SAMPLING_FREQUENCY)
        
    for data in data_loader:
        assert data.shape == (
            len(config.bands),
            config.sample_length * config.new_fs,
            8,
            8,
        )


def test_data_loader_grid_creation_returns_input_data(data_loader_creation_fn):
    config = ECoGDataConfig(
        batch_size=32, bands=[[4, 8], [8, 13], [13, 30], [30, 55]], new_fs=20,
    )
    fake_data = create_fake_sin_data()
    data_loader = data_loader_creation_fn(config, data=fake_data, file_sampling_frequency=FILE_SAMPLING_FREQUENCY)
    
    assert np.allclose(
        data_loader._load_grid_data(),
        fake_data[
            :64, :
        ],
    )


def test_data_loader_can_handle_missing_channel(data_loader_creation_fn):
    config = ECoGDataConfig(
        batch_size=32, bands=[[4, 8], [8, 13], [13, 30], [30, 55]], new_fs=20,
    )
    # Omit "G1" from list of channels so loader will fill that channels with zeros.
    ch_names = ["G" + str(i + 1) for i in range(1, 65)]
    # To make checking easier just make data all ones.
    fake_data = np.ones((len(ch_names), 100 * FILE_SAMPLING_FREQUENCY))
    data_loader = data_loader_creation_fn(config, ch_names=ch_names, data=fake_data, file_sampling_frequency=FILE_SAMPLING_FREQUENCY)

    actual_data = data_loader._load_grid_data()
    assert np.all(np.isnan(actual_data[0]))
    assert np.allclose(
        actual_data[1:],
        np.ones((63, 100 * FILE_SAMPLING_FREQUENCY)),
    )
    assert actual_data.dtype == np.float32
        

def test_data_loader_drops_short_signals(data_loader_creation_fn):
    config = ECoGDataConfig(
        batch_size=32, bands=[[4, 8], [8, 13], [13, 30], [30, 55]], new_fs=20, sample_length=1
    )
    # Create data for 10.5 seconds.
    fake_data = np.ones((65, int(10.5 * FILE_SAMPLING_FREQUENCY)))
    data_loader = data_loader_creation_fn(config, data=fake_data, file_sampling_frequency=FILE_SAMPLING_FREQUENCY)

    for i, _ in enumerate(data_loader):
        pass
    
    assert i == 9

def test_data_loader_resets_from_beginning_of_dataset(data_loader_creation_fn):
    config = ECoGDataConfig(
        batch_size=32, bands=[[4, 8], [8, 13], [13, 30], [30, 55]], new_fs=20, sample_length=1
    )
    # Make sure we can iterate through data twice and it returns the same data each time.
    data_loader = data_loader_creation_fn(config, data=create_fake_sin_data(), file_sampling_frequency=FILE_SAMPLING_FREQUENCY)
    
    data_first_pass = []
    for first_pass_data in data_loader:
        data_first_pass.append(first_pass_data)
    
    i = 0
    for i, second_pass_data in enumerate(data_loader):
        assert np.all(second_pass_data == data_first_pass[i])
        
    # Make sure it actually iterated a second time.
    assert i > 0
