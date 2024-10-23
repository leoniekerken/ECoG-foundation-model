import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    # Task Config.
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to saved model checkpoint used for generating neural embeddings.",
    )
    parser.add_argument(
        "--embedding-device",
        type=str,
        default="cpu",
        help="What device to run the embedding generation on. Likely one of cuda or cpu.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=1,
        help="Batch size to use when generating neural embeddings.",
    )
    
    # Data config.
    parser.add_argument(
        "--lag",
        type=int,
        default=0,
        help="Determines where relative to the word onset in ms the neural signal should be read from. Lag = 0 means neural signal will be read starting at word onset. Lag = -2000 means neural signal will be read starting at 2 seconds before word onset. Lag = 100 means neural signal will be read starting at 100 ms before word onset.",
    )
    
    parser.add_argument(
        "--electrode-glob-path",
        type=str,
        default="NY*_*_Part*_conversation*_electrode_preprocess_file_{elec_id}.mat",
        help="The glob path to access electrode data within encoding_neural_data_folder. Should include placeholder for electrode_id {elec_id} See default for sample value."
    )
    
    parser.add_argument(
        "--encoding-neural-data-folder",
        type=str,
        help="Path to parent directory of electrode data to be used in encoding."
    )
    
    parser.add_argument(
        "--conversation-data-df-path",
        type=str,
        help="Path to the dataframe containing conversation data (i.e. embeddings, word onsets and offsets, etc)."
    )

    args = parser.parse_args()

    return args
