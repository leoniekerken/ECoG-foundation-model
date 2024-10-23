import torch
from einops import rearrange


def get_padding_mask(signal, model, device):
    """
    Zero padding for channels that were rejected during preprocessing for bad signal quality

    Args:
        signal: torch tensor of shape batch size * number of bands * timepoints * d * h * w
        model: model object
        device: GPU device

    Returns:
        padding_mask: boolean tensor of same shape as patchified signal indicating which parts of the signal are padded

    """
    padding_mask = ~torch.isnan(signal).to(device)
    padding_mask = rearrange(model.patchify(padding_mask), "b ... d -> b (...) d")

    # TODO make flexible for handling ps > 1
    padding_mask = torch.all(padding_mask, dim=0)
    padding_mask = torch.all(padding_mask, dim=1)

    return padding_mask


def get_tube_mask(tube_mask_ratio, height, width, padding_mask, device):

    """
    Masking out a certain percentage of the original signal, the unmasked parts are fed into the encoder.
    When constructing the mask we are taking into account channels that are padded, such that only channels
    with actual data are not masked out. Channels that were rejected are automatically masked out.

    Args:
        tube_mask_ratio: Proportion of tubes to mask out.
        height: height of electrode grid
        width: width of electrode grid
        padding_mask: boolean tensor indicating which channels contain data
        device: GPU device

    Returns:
        tube_mask: boolean tensor of shape as patchified signal indicating which parts are fed into the encoder
        (True) and which not (False)
    """

    # construct a tube mask of size number of channels
    tube_mask = (
        torch.zeros(height * width)
        .to(device)
        .to(torch.bool)
    )

    # only select idx of existing channels here - the first len(number of channels) entries in the padding mask
    # indicate whether the channel contains a signal or not and we are only taking those which contain a signal (True) -
    # chn_idx contains the indices of channels with data
    chn_idx = torch.nonzero(padding_mask[: len(tube_mask)])

    # shuffling values in chn_idx
    mask_idx_candidates = chn_idx[torch.randperm(len(chn_idx))]

    # now we are taking 1 - tube_mask_ratio percent of all (shuffled) channels that contain signal
    tube_idx = mask_idx_candidates[
        : int(
            len(mask_idx_candidates)
            * (1 - tube_mask_ratio)
        )
    ]
    # and set them to True, meaning they will be unmasked
    tube_mask[tube_idx] = True

    # now we repeat this pattern of masking across the remaining patches
    tube_mask = tube_mask.tile(len(padding_mask) // height // width)

    return tube_mask


def get_decoder_mask(decoder_mask_ratio, tube_mask, device):
    """
    Getting parts of the signal that were not seen by the encoder to be reconstructed by the decoder. Additionally,
    args.decoder_mask_ratio != 0, we also mask out parts of the remaining unsee signal to reduce computational cost.

    Args:
        decoder_mask_ratio: The ratio of the number of masked tokens in the input sequence
        tube_mask: boolean tensor indicating which parts of the patchified signal are masked out for the encoder
        device: GPU device

    Returns:
        decoder_mask: boolean tensor of shape as patchified signal indicating which of the patches are masked
            for the decoder.
    """

    decoder_mask = torch.zeros_like(tube_mask).to(device).to(torch.bool)
    remaining_mask_idx = (~tube_mask).nonzero()
    decoder_mask_idx = remaining_mask_idx[
        : int(len(remaining_mask_idx) * (1 - decoder_mask_ratio))
    ]
    decoder_mask[decoder_mask_idx] = True

    return decoder_mask


def get_running_cell_mask(decoder_mask_ratio, frame_patch_size, num_frames, tube_mask, device):
    """
    Not implemented for now
    """

    num_patch_per_cell = 4
    num_mask_per_cell = int(decoder_mask_ratio * num_patch_per_cell)
    stride = int(num_patch_per_cell / num_mask_per_cell)
    num_patch_per_frame = 16  # change to be flexible #TODO
    cell = torch.ones(num_frames // frame_patch_size, num_patch_per_cell)
    # mask out patches in cell so that the mask spatially progresses across frames
    # Quing et al., 2023, MAR: Masked Autoencoder for Efficient Action Recognition
    for i in range(num_frames // frame_patch_size):
        for j in range(num_mask_per_cell):
            cell[i, (int(j * stride) + i) % num_patch_per_cell] = 0

    running_cell_mask = (
        cell.repeat(1, int(num_patch_per_frame / num_patch_per_cell))
        .flatten(0)
        .to(device)
        .to(torch.bool)
    )

    # filter out patches that were seen by the encoder
    running_cell_mask[~tube_mask] == False

    return running_cell_mask
