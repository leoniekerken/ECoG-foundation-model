import os

import numpy as np
import torch

# Needs to be included for model loading.
from models import SimpleViT
# Needs to be included for config loading.
from config import create_video_mae_experiment_config_from_config_file
from downstream_tasks.encoding_decoding.parser import arg_parser
from downstream_tasks.encoding_decoding.config import create_encoding_decoding_experiment_config
from downstream_tasks.encoding_decoding.utils import (
    run_decoding_task
)


def main(args):
    # Setup config
    experiment_config = create_encoding_decoding_experiment_config(args)
    inference_device_name = experiment_config.encoding_task_config.embedding_device

    # Load model
    # Needed to load these classes into safe globals if we want to do torch.load with weights_only.
    # torch.load without weights_only uses pickle which is unsafe and can run arbitrary code
    # if you're not careful.
    torch.serialization.add_safe_globals([SimpleViT])
    checkpoint = torch.load(
        experiment_config.encoding_task_config.model_path,
        map_location={"cuda": inference_device_name, "cpu": inference_device_name},
        weights_only=True,
    )
    model = checkpoint["model"]
    model.device = inference_device_name

    ecog_data_config = create_video_mae_experiment_config_from_config_file(os.path.join(experiment_config.encoding_task_config.model_path, "experiment_config.ini"))
    rp, mspe = run_decoding_task(experiment_config, ecog_data_config, model)

    # TODO: Improve metrics used to measure performance of encoding task.
    print("Pearson correlations:", rp)
    print("MSPE:", mspe)


if __name__ == "__main__":
    args = arg_parser()
    main(args)
