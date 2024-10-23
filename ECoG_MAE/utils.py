import os
import torch
import numpy as np
import random
import scipy
from einops import rearrange
from typing import Optional

from config import ECoGDataConfig, ViTConfig


def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# TODO: Test this function.
def preprocess_neural_data(
    signal: np.array,
    fs: int,
    new_fs: int,
    sample_secs: int,
    bands: Optional[list[list[int]]] = None,
    norm: Optional[str] = None,
    means: Optional[np.array] = None,
    stds: Optional[np.array] = None,
    pad_before_sample: bool = False,
    dtype=np.float32,
) -> np.array:
    """Preprocess and reshape neural data for VideoMAE model.

    Args:
        signal (np.array): Of shape [num_electrodes, num_samples]. Should already be cropped to the desired number of samples.
        fs (int): The sampling rate of the signal.
        new_fs (int): The sampling rate to resample the data to.
        sample_secs (int): The number of seconds in a sample.
        bands (Optional[list[list[int]]], optional): Should be a list of the form [[4, 8], [10, 50]] where
            each set of two numbers represents a band of frequencies to filter from the provided signal.
            If not set then signal is assumed to represent one band and is used as a lone band signal.
            Defaults to None.
        norm (Optional[str], optional): If "hour" then will use the passed means and stds to normalize the signal. Defaults to None.
        means (Optional[np.array], optional): Of shape [num_electrodes]. Means for each electrode. Defaults to None.
        stds (Optional[np.array], optional): Of shape [num_electrodes]. Standard deviations for each electrode. Defaults to None.
        pad_before_sample (bool): If true then samples which are not the desired length will be padded with 0's before the actual extracted signal. Useful if sample is taken from the very start of the signal.

    Returns:
        np.array:
            shape c*t*d*h*w, where
            c = freq bands,
            t = number of datapoints within a sample
            d = depth (currently 1)
            h = height of grid (currently 8)
            w = width of grid (currently 8)
    """

    def norm(input, ch_idx):
        output = input - means[ch_idx] / stds[ch_idx]

        return output

    if norm == "hour":

        # z-score signal for each channel separately
        for i in range(0, 64):
            signal[i, :] = norm(signal[i], i)

    # Extract frequency bands if provided.
    if bands:
        filtered_signal = np.zeros((len(bands), 64, signal.shape[1]))

        for i, freqs in enumerate(bands):
            sos = scipy.signal.butter(
                4, freqs, btype="bandpass", analog=False, output="sos", fs=fs
            )
            filtered_signal[i] = scipy.signal.sosfiltfilt(sos, signal)
            filtered_signal[i] = np.abs(scipy.signal.hilbert(filtered_signal[i]))
    else:
        # Add band axis of size 1 for non-filtered data.
        filtered_signal = np.expand_dims(signal, axis=0)

    resampled = resample_mean_signals(filtered_signal, fs, new_fs)
    # rearrange into shape c*t*d*h*w, where
    # c = freq bands,
    # t = number of datapoints within a sample
    # d = depth (currently 1)
    # h = height of grid (currently 8)
    # w = width of grid (currently 8)
    preprocessed_signal = rearrange(
        np.array(resampled, dtype=np.float32), "c (h w) t -> c t () h w", h=8, w=8
    )

    # Zero-pad if sample is too short.
    expected_sample_length = sample_secs * new_fs
    if preprocessed_signal.shape[1] < expected_sample_length:
        padding = np.zeros(
            (
                preprocessed_signal.shape[0],
                expected_sample_length - preprocessed_signal.shape[1],
                1,
                8,
                8,
            ),
            dtype=dtype,
        )
        if pad_before_sample:
            preprocessed_signal = np.concatenate((padding, preprocessed_signal), axis=1)
        else:
            preprocessed_signal = np.concatenate((preprocessed_signal, padding), axis=1)

    return preprocessed_signal


def resample_mean_signals(signal: np.array, old_fs: int, new_fs: int) -> np.array:
    """Resample signal with sampling rate of old_fs Hz to new_fs Hz by taking means over windows of data.

    Args:
        signal (np.array): shape [bands, num_electrodes, samples]
        old_fs (int): Old sample rate in Hz.
        new_fs (int): Sample rate to resample to in Hz.

    Returns:
        np.array: Resampled signal with the new sample rate.
    """
    window_width = old_fs / new_fs
    num_samples = signal.shape[2]
    # TODO: revisit using ceil here. By using ceil our final entry in our new array may be averaged
    # over fewer samples then the previous entries.
    num_new_samples = int(np.ceil(num_samples * new_fs / old_fs))

    resampled_signal = np.zeros((signal.shape[0], signal.shape[1], num_new_samples))

    # Current index needs to be a float to handle cases where old_fs and new_fs are not divisible.
    # Flooring results only after summing window widths ensures all of the data is included in the
    # averages.
    current_idx = 0.0
    for i in range(num_new_samples):
        start_idx = int(np.floor(current_idx))
        end_idx = int(np.floor(current_idx + window_width))

        if end_idx > start_idx:  # Ensure there is a range to average over
            resampled_signal[:, :, i] = np.mean(signal[:, :, start_idx:end_idx], axis=2)
        else:  # If the window size rounds to 0, just take the value at start_idx
            resampled_signal[:, :, i] = signal[:, :, start_idx]

        current_idx += window_width

    return resampled_signal


def get_signal_stats(signal: np.array) -> tuple[np.array, np.array]:
    """Generate means and standard deviations for all electrodes in signal.

    Args:
        signal (np.array): Shape [num_electrodes, num_samples].

    Returns:
        tuple[np.array, np.array]: (means, stds) each array has shape [num_electrodes].
    """
    return np.mean(signal, axis=1), np.std(signal, axis=1)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("param counts:\n{:,} total\n{:,} trainable".format(total, trainable))
    return trainable


def normalize(raw_signal):

    mean = torch.mean(raw_signal, dim=(0, 2), keepdim=True)
    std = torch.std(raw_signal, dim=(0, 2), keepdim=True)
    signal = (raw_signal - mean) / std

    return signal


def get_signal(
    patches: torch.Tensor,
    batch_size: int,
    num_bands: int,
    num_frames: int,
    model_config: ViTConfig,
) -> torch.Tensor:
    """Convert patches into a signal of shape [electrodes, num_bands, num_frames]

    Args:
        patches (torch.Tensor): Patch transformed signal of model.
        batch_size (int): The number of examples in a batch.
        num_bands (int): Number of bands in patches.
        num_frames (int): Number of frames in patches.
        model_config (ViTConfig): Config for model.
    """
    return rearrange(
        patches,
        "b (f d s) (pd ph pw pf c) -> b (d pd s ph pw) c (f pf)",
        c=num_bands,
        d=1,
        f=num_frames // model_config.frame_patch_size,
        pd=model_config.patch_dims[0],
        ph=model_config.patch_dims[1],
        pw=model_config.patch_dims[2],
        pf=model_config.frame_patch_size,
    )


def rearrange_signals(
    model_config: ViTConfig,
    model,
    device,
    signal,
    decoder_out,
    padding_mask,
    tube_mask,
    decoder_mask,
    decoder_padding_mask,
):
    batch_size = signal.shape[0]
    num_bands = signal.shape[1]
    num_frames = signal.shape[2]

    # parts of the reconstructed signal that were not seen by the encoder
    unseen_output = decoder_out[:, len(tube_mask.nonzero()) :]

    # parts of the reconstructed signal that were seen by the encoder
    seen_output = decoder_out[:, : len(tube_mask.nonzero())]

    # rearrange original signal into patches
    target_patches = model.patchify(signal)
    target_patches_vit = rearrange(target_patches, "b ... d -> b (...) d")

    # parts of the original signal not seen by the encoder
    unseen_target = target_patches_vit[:, decoder_mask][:, decoder_padding_mask]

    # parts of the original signal seen by the encoder
    seen_target = target_patches_vit[:, ~decoder_mask]

    # rearranging seen and unseen parts of the reconstructed signal into original position
    full_recon_patches = (
        torch.zeros(target_patches_vit.shape).fill_(float("nan")).to(device)
    )

    tube_idx = torch.nonzero(tube_mask).squeeze()
    decoder_idx = torch.nonzero(decoder_mask & padding_mask).squeeze()

    full_recon_patches[:, tube_idx, :] = seen_output
    full_recon_patches[:, decoder_idx, :] = unseen_output

    full_recon_signal = get_signal(
        full_recon_patches, batch_size, num_bands, num_frames, model_config
    )

    full_target_signal = rearrange(signal, "b c f d h w -> b (h w d) c f")

    # rearrange unseen patches into signal
    unseen_recon_signal = get_signal(
        unseen_output, batch_size, num_bands, num_frames, model_config
    )

    unseen_target_signal = get_signal(
        unseen_target, batch_size, num_bands, num_frames, model_config
    )

    return (
        full_recon_signal,
        full_target_signal,
        unseen_output,
        unseen_target,
        seen_output,
        seen_target,
        unseen_target_signal,
        unseen_recon_signal,
    )


def contrastive_loss(
    cls_token1: torch.Tensor, cls_token2: torch.Tensor, temperature: torch.Tensor
):
    feat1 = cls_token1 / cls_token1.norm(dim=1, keepdim=True)
    feat2 = cls_token2 / cls_token2.norm(dim=1, keepdim=True)

    cosine_sim = feat1 @ feat2.T
    logit_scale = temperature.exp()  # log scale, learned during training
    feat1 = cosine_sim * logit_scale
    feat2 = feat1.T

    labels = torch.arange(feat1.shape[0]).to(feat1.device)
    loss = (
        torch.nn.functional.cross_entropy(feat1, labels)
        + torch.nn.functional.cross_entropy(feat2, labels)
    ) / 2
    return loss


def get_signal_correlations(signal_a, signal_b):
    """Get correlation coefficients between channels of signal_a and signal_b averaged over batch.

    Args:
        signal_a (tensor): shape [batch, num_electrodes, channels, observations]
        signal_b (tensor): shape [batch, num_electrodes, channels, observations]
        
    Returns:
        tensor of shape [num_electrodes, channels] where each entry is the correlation found between
        the two signals for that electrode and channel.
    """
    correlation_matrix = torch.zeros(
        signal_a.shape[0], signal_a.shape[1], signal_a.shape[2]
    ).fill_(torch.nan)
    
    for batch in range(signal_a.shape[0]):
        for electrode in range(signal_a.shape[1]):
            for channel in range(signal_a.shape[2]):
                correlation_matrix[batch, electrode, channel] = torch.corrcoef(
                    torch.stack([
                        signal_a[batch, electrode, channel],
                        signal_b[batch, electrode, channel],
                    ])
                )[0, 1]
                
    return correlation_matrix.mean(dim=0)
