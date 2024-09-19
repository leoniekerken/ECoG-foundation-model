# Load model,
# create dataloader
# iterate through dataloader and generate embeddings
# train linear layer

import numpy as np
import torch

# Needs to be included for model loading.
from models import SimpleViT
# Needs to be included for config loading.
from config import VideoMAEExperimentConfig
from downstream_tasks.encoding.load_signal import EncodingDataset
from downstream_tasks.encoding.parser import arg_parser
from downstream_tasks.encoding.config import create_encoding_experiment_config
from downstream_tasks.encoding.utils import (
    pearson_correlation,
    run_regression,
    generate_embedding_dataset,
    merge_data_configs,
)


def main(args):
    # Setup config
    experiment_config = create_encoding_experiment_config(args)
    inference_device_name = experiment_config.encoding_task_config.embedding_device

    # Load model
    # Needed to load these classes into safe globals if we want to do torch.load with weights_only.
    # torch.load without weights_only uses pickle which is unsafe and can run arbitrary code
    # if you're not careful.
    torch.serialization.add_safe_globals([SimpleViT, VideoMAEExperimentConfig])
    checkpoint = torch.load(
        experiment_config.encoding_task_config.model_path,
        map_location={"cuda": inference_device_name, "cpu": inference_device_name},
        weights_only=True,
    )
    model = checkpoint["model"]
    model.device = inference_device_name

    ecog_data_config = checkpoint["experiment_config"].ecog_data_config
    encoding_data_config = merge_data_configs(experiment_config.encoding_data_config, ecog_data_config)

    dataset = EncodingDataset(encoding_data_config)

    word_embeddings, neural_embeddings = generate_embedding_dataset(
        dataset,
        model,
        experiment_config.encoding_task_config.embedding_batch_size,
        inference_device_name,
    )

    predictions = run_regression(
        word_embeddings,
        neural_embeddings,
        experiment_config.encoding_task_config.num_folds,
    )

    rp, _, _ = pearson_correlation(neural_embeddings, predictions)
    mspe = np.square(neural_embeddings - predictions).mean()

    # TODO: Improve metrics used to measure performance of encoding task.
    print("Pearson correlations:", rp)
    print("MSPE:", mspe)


if __name__ == "__main__":
    args = arg_parser()
    main(args)