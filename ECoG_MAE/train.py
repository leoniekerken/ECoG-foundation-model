import multiprocessing as mp
import time as t
import logging
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import VideoMAEExperimentConfig, write_config_file
from pretrain_utils import train_single_epoch, test_single_epoch


logger = logging.getLogger(__name__)


def train_model(
    config: VideoMAEExperimentConfig,
    device: str,
    model,
    train_dl: torch.utils.data.DataLoader,
    test_dl: torch.utils.data.DataLoader,
    optimizer,
    lr_scheduler,
    accelerator,
    data_type,
    local_rank,
):
    """
    Runs model training

    Args:
        config: experiment config for this run
        device: the gpu to be used for model training
        model: an untrained model instance with randomly initialized parameters
        train_dl: dataloader instance for train split
        test_dl: dataloader instance for test split
        num_patches: number of patches in which the input data is segmented
        optimizer: Adam optimizer instance - https://www.analyticsvidhya.com/blog/2023/12/adam-optimizer/
        lr_scheduler: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        accelerator: an accelerator instance - https://huggingface.co/docs/accelerate/en/index
        data_type: the data type to be used, we use "fp16" mixed precision - https://towardsdatascience.com/understanding-mixed-precision-training-4b246679c7c4
        local_rank: the local rank environment variable (only needed for multi-gpu training)

    Returns:
        model: model instance with updated parameters after training
    """
    

    torch.cuda.empty_cache()
    model.to(device)

    num_frames = config.ecog_data_config.sample_length * config.ecog_data_config.new_fs

    # if config.trainer_config.learning_rate == None:
    #     model, optimizer, train_dl, lr_scheduler = accelerator.prepare(
    #         model, optimizer, train_dl, lr_scheduler
    #     )
        
    os.makedirs(config.logging_config.event_log_dir, exist_ok=True)
    # TODO: Make this less likely to cause accidental overwrites.
    log_writer = SummaryWriter(log_dir=os.path.join(config.logging_config.event_log_dir, config.job_name))
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", config.job_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    write_config_file(os.path.join(checkpoint_dir, "experiment_config.ini"), config)

    mse = nn.MSELoss()

    for epoch in range(config.trainer_config.num_epochs):
        start = t.time()
        with torch.cuda.amp.autocast(dtype=data_type):
            model.train()
            train_single_epoch(train_dl,
                               epoch,
                               accelerator,
                               optimizer,
                               lr_scheduler,
                               device,
                               model,
                               config,
                               num_frames,
                               logger,
                               mse,
                               log_writer=log_writer)

            test_single_epoch(test_dl, epoch, device, model, config, num_frames, logger, mse, log_writer=log_writer)

            end = t.time()

            logger.info("Epoch " + str(epoch) + " done. Time elapsed: " + str(end - start))

        # save model checkpoints
        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
            
        # Save a different checkpoint for every epoch.
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"{epoch}_checkpoint.pth"))

    return model
