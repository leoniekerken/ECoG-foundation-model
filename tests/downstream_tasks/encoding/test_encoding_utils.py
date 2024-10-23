from config import ECoGDataConfig
from downstream_tasks.encoding_decoding.utils import merge_data_configs
from downstream_tasks.encoding_decoding.config import EncodingDecodingDataConfig


def test_merge_data_configs_correctly_sets_ecog_config_values():
    ecog_data_config = ECoGDataConfig(
        norm = "norm",
        data_size = 0.9,
        batch_size = 16,
        env = True,
        bands = [[3, 4]],
        original_fs = 123,
        new_fs = 20,
        dataset_path = "fake_path",
        train_data_proportion = 0.5,
        sample_length = 4,
        shuffle = True,
        test_loader = True,
    )
    
    encoding_data_config = EncodingDecodingDataConfig(
        conversation_data_df_path = "conversation_path",
        encoding_neural_data_folder = "neural_folder",
        electrode_glob_path = "NY*_*_Part*_conversation*_electrode_preprocess_file_{elec_id}.mat",
        lag = 3,
        original_fs = 512,
    )
    
    final_config = merge_data_configs(encoding_data_config, ecog_data_config)
    
    expected_final_config = EncodingDecodingDataConfig(
        conversation_data_df_path = "conversation_path",
        encoding_neural_data_folder = "neural_folder",
        electrode_glob_path = "NY*_*_Part*_conversation*_electrode_preprocess_file_{elec_id}.mat",
        lag = 3,
        original_fs = 512,
        norm = "norm",
        data_size = 0.9,
        batch_size = 16,
        env = True,
        bands = [[3, 4]],
        new_fs = 20,
        dataset_path = "fake_path",
        train_data_proportion = 0.5,
        sample_length = 4,
        shuffle = True,
        test_loader = True,
    )
    
    assert final_config == expected_final_config
    