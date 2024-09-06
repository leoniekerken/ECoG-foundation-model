from dataclasses import dataclass, field

from config import ECoGDataConfig

@dataclass
class EncodingDataConfig(ECoGDataConfig):
    # Path to the dataframe containing conversation data (i.e. embeddings, word onsets and offsets, etc)
    conversation_data_df_path: str = ""
    
    # Path to parent directory of electrode data to be used in encoding.
    encoding_neural_data_folder: str = ""
    
    # The glob path to access electrode data within encoding_neural_data_folder. Should include placeholder for electrode_id {elec_id} See default for sample value.
    # TODO: In the future include options for subject id and conv id but keeping it simple for now with just electrode id.
    electrode_glob_path: str = "NY*_*_Part*_conversation*_electrode_preprocess_file_{elec_id}.mat"

    # Determines where relative to the word onset in ms the neural signal should be read from.
    # Lag = 0 means neural signal will be read starting at word onset.
    # Lag = -2000 means neural signal will be read starting at 2 seconds before word onset.
    # Lag = 100 means neural signal will be read starting at 100 ms before word onset.
    lag: int = 0
    
@dataclass
class EncodingTaskConfig:
    # Path to the model checkpoint used to generate neural embeddings.
    # Assumes model was saved using torch.save and is the SimpleViT in models.py.
    model_path: str = ""
    
    # What device to run the embedding generation on. Likely one of "cuda" or "cpu"
    embedding_device: str = "cpu"
    
    # Batch size to use when generating neural embeddings.
    embedding_batch_size: int = 1
    
    # The number of folds to use when training the encoder.
    num_folds: int = 2
    

@dataclass
class EncodingExperimentConfig:
    encoding_data_config: EncodingDataConfig = field(default_factory=EncodingDataConfig)
    encoding_task_config: EncodingTaskConfig = field(default_factory=EncodingTaskConfig)


def create_encoding_experiment_config(args) -> EncodingExperimentConfig:
    """Convert command line arguments to an experiment config for VideoMAE."""
    return EncodingExperimentConfig(
        encoding_task_config=EncodingTaskConfig(
            model_path=args.model_path,
            embedding_device=args.embedding_device,
        ),
        encoding_data_config=EncodingDataConfig(
            # ECoGDataConfig variables are loaded from the model checkpoint to ensure data is preprocessed in the same way.
            encoding_neural_data_folder=args.encoding_neural_data_folder,
            conversation_data_df_path=args.conversation_data_df_path,
            electrode_glob_path=args.electrode_glob_path,
            lag=args.lag,
        ),
    )