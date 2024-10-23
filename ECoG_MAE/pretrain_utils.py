import torch
from torch.utils.data import DataLoader
from mask import *
from utils import *
from metrics import *
from plot import *
import constants
from models import SimpleViT
from config import VideoMAEExperimentConfig

import mae_st_util.misc as misc
from mae_st_util.logging import master_print as print


def forward_model(signal, model, device, config, num_frames, mse, calculate_correlations=False):
    # TODO: Actually move this code into model.
    model_config = config.video_mae_task_config.vit_config
    padding_mask = get_padding_mask(signal, model, device)

    # convert nans to 0
    signal = torch.nan_to_num(signal)

    # masking out parts of the input to the encoder (same mask across frames)
    tube_mask = get_tube_mask(
        config.video_mae_task_config.tube_mask_ratio,
        constants.GRID_HEIGHT,
        constants.GRID_WIDTH,
        padding_mask,
        device,
    )

    # selecting parts of the signal for the decoder to reconstruct
    if config.video_mae_task_config.decoder_mask_ratio == 0:
        decoder_mask = get_decoder_mask(
            config.video_mae_task_config.decoder_mask_ratio,
            tube_mask,
            device,
        )
    else:
        if config.video_mae_task_config.running_cell_masking:
            decoder_mask = get_running_cell_mask(
                config.video_mae_task_config.decoder_mask_ratio,
                model_config.frame_patch_size,
                num_frames,
                tube_mask,
                device,
            )

    # make sure the decoder only reconstructs channels that were not discarded during preprocessing
    decoder_padding_mask = padding_mask[decoder_mask]

    # encode the tube patches
    encoder_out = model(signal, encoder_mask=tube_mask, tube_padding_mask=padding_mask)

    # decode both the encoder_out patches and masked decoder patches
    decoder_out = model(
        encoder_out,
        encoder_mask=tube_mask,
        decoder_mask=decoder_mask,
        decoder_padding_mask=decoder_padding_mask,
    )

    # rearrange reconstructed and original patches (seen and not seen by encoder) into signal
    (
        full_recon_signal,
        full_target_signal,
        unseen_output,
        unseen_target,
        seen_output,
        seen_target,
        unseen_target_signal,
        unseen_recon_signal,
    ) = rearrange_signals(
        model_config,
        model,
        device,
        signal,
        decoder_out,
        padding_mask,
        tube_mask,
        decoder_mask,
        decoder_padding_mask,
    )

    # calculate loss
    loss = mse(unseen_output, unseen_target)
    seen_loss = mse(seen_output, seen_target)

    # TODO: improve efficiency so we can call this everytime.
    if calculate_correlations:
        # get correlations
        total_signal_correlations = get_signal_correlations(
            full_recon_signal, full_target_signal
        )
        unseen_signal_correlations = get_signal_correlations(
            unseen_recon_signal, unseen_target_signal
        )

        return loss, seen_loss, total_signal_correlations, unseen_signal_correlations
    else:
        return loss, seen_loss


def train_single_epoch(
    train_dl: DataLoader,
    epoch: int,
    accelerator,
    optimizer,
    lr_scheduler,
    device: str,
    model: SimpleViT,
    config: VideoMAEExperimentConfig,
    num_frames: int,
    logger,
    mse,
    log_writer=None,
):
    model.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)

    for train_i, batch in enumerate(
        metric_logger.log_every(train_dl, config.logging_config.print_freq, header)
    ):
        optimizer.zero_grad()

        signal = batch.to(device)

        if config.ecog_data_config.norm == "batch":
            signal = normalize(signal)
        else:
            signal = torch.where(signal == 0, torch.tensor(float("nan")), signal)

        # mask indicating positions of channels that were rejected during preprocessing
        loss, seen_loss = (
            forward_model(signal, model, device, config, num_frames, mse)
        )
        if torch.isnan(loss):
            logger.error(
                f"Got nan loss for index {train_i}. Ignoring and continuing..."
            )
            continue

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        loss_value = loss.item()
        seen_loss_value = seen_loss.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if log_writer is not None:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((train_i / len(train_dl) + epoch) * 1000)
            log_writer.add_scalar("loss/train", loss_value, epoch_1000x)
            log_writer.add_scalar("loss/train_seen", seen_loss_value, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test_single_epoch(
    test_dl: DataLoader,
    epoch: int,
    device: str,
    model: SimpleViT,
    config: VideoMAEExperimentConfig,
    num_frames: int,
    logger,
    mse,
    log_writer=None,
):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        running_seen_loss = 0.0
        running_total_signal_correlation = 0.0
        running_unseen_signal_correlation = 0.0
        for test_i, batch in enumerate(test_dl):
            signal = batch.to(device)

            if config.ecog_data_config.norm == "batch":
                signal = normalize(signal)
            else:
                signal = torch.where(signal == 0, torch.tensor(float("nan")), signal)

            loss, seen_loss, total_signal_correlations, unseen_signal_correlations = (
                forward_model(signal, model, device, config, num_frames, mse, calculate_correlations=True)
            )
            if torch.isnan(loss):
                logger.error(
                    f"Got nan loss for index {test_i}. Ignoring and continuing..."
                )
                continue

            running_loss += loss.item()
            running_seen_loss += seen_loss.item()
            running_total_signal_correlation += total_signal_correlations.mean().item()
            running_unseen_signal_correlation += (
                unseen_signal_correlations.mean().item()
            )

        loss_mean = running_loss / (test_i + 1)
        seen_loss_mean = running_seen_loss / (test_i + 1)
        total_signal_correlation_mean = running_total_signal_correlation / (test_i + 1)
        unseen_signal_correlation_mean = running_unseen_signal_correlation / (
            test_i + 1
        )

        # Write averages for test data.
        if log_writer is not None:
            """We use epoch_1000x as the x-axis in tensorboard.
            This aligns with the training loop.
            """
            epoch_1000x = int((test_i / len(test_dl) + epoch) * 1000)
            log_writer.add_scalar("loss/test", loss_mean, epoch_1000x)
            log_writer.add_scalar("loss/test_seen", seen_loss_mean, epoch_1000x)
            log_writer.add_scalar(
                "correlation/test_total_signal",
                total_signal_correlation_mean,
                epoch_1000x
            )
            log_writer.add_scalar(
                "correlation/test_unseen_signal",
                unseen_signal_correlation_mean,
                epoch_1000x
            )
