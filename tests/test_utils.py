import pytest

from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import torch

from config import ViTConfig
from mask import get_tube_mask, get_decoder_mask
from utils import resample_mean_signals, rearrange_signals, get_signal_correlations

FRAME_PATCH_SIZE = 4
NUM_BANDS = 5
GRID_HEIGHT = 8
GRID_WIDTH = 8
NUM_FRAMES = 40

@pytest.fixture
def fake_model():
    class FakeModel():
        def __init__(self):
            self.patchify = Rearrange(
                "b c (f pf) (d pd) (h ph) (w pw) -> b f d h w (pd ph pw pf c)",
                pd=1,
                ph=1,
                pw=1,
                pf=FRAME_PATCH_SIZE,
            )

    return FakeModel()

def test_resampling_correctly_averages_windows():
    # Signal of shape [bands, electrodes, samples]
    input_signal = np.array(
        [
            [[1, 1, 1, 1, 2, 2, 2, 2], [3, 3, 3, 3, 4, 4, 4, 4]],
            [[5, 5, 5, 5, 6, 6, 6, 6], [7, 7, 7, 7, 8, 8, 8, 8]],
        ]
    )

    old_fs = 4
    new_fs = 1

    assert np.allclose(
        resample_mean_signals(input_signal, old_fs, new_fs),
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    )
   
    
def test_resampling_can_handle_not_perfectly_divisible_durations():
    # Signal of shape [bands, electrodes, samples]
    input_signal = np.array(
        [
            [[1, 1, 1, 1, 2, 2, 2, 2, 10], [3, 3, 3, 3, 4, 4, 4, 4, 11]],
            [[5, 5, 5, 5, 6, 6, 6, 6, 12], [7, 7, 7, 7, 8, 8, 8, 8, 13]],
        ]
    )

    old_fs = 4
    new_fs = 1

    assert np.allclose(
        resample_mean_signals(input_signal, old_fs, new_fs),
        np.array([[[1, 2, 10], [3, 4, 11]], [[5, 6, 12], [7, 8, 13]]]),
    )
    
def test_resampling_can_handle_non_divisible_sampling_rates():
    # Signal of shape [bands, electrodes, samples]
    input_signal = np.array(
        [
            # This is not an  optimal test setup because it assumes the implementation of the function
            # but suffices for now. This could be made better in the future.
            [[1, 2, 3, 3], [4, 5, 6, 6]],
            [[7, 8, 9, 9], [10, 11, 12, 12]],
        ]
    )

    old_fs = 4
    new_fs = 3

    assert np.allclose(
        resample_mean_signals(input_signal, old_fs, new_fs),
        np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
    )

def test_rearrange_signals_without_padding(fake_model):
    device="cpu"
    simple_fake_signal = torch.randn(16, GRID_HEIGHT *  GRID_WIDTH, NUM_BANDS, NUM_FRAMES)
    # Rearrange for model inputs.
    fake_signal = rearrange(simple_fake_signal, "b (h w) c f -> b c f 1 h w", h=GRID_HEIGHT, w=GRID_WIDTH)
    model_config = ViTConfig(frame_patch_size=4, patch_size=1)
    
    patched_signal = fake_model.patchify(fake_signal).view(16, -1, 20)
    padding_mask = torch.ones(patched_signal.shape[1], dtype=torch.bool)
    encoder_mask = get_tube_mask(0.5, GRID_HEIGHT, GRID_WIDTH, padding_mask, "cpu")
    decoder_mask = get_decoder_mask(0., encoder_mask, "cpu")
    decoder_padding_mask = padding_mask[decoder_mask]

    model_seen_output = patched_signal[:, encoder_mask, :]
    model_masked_output = patched_signal[:, decoder_mask, :]
    
    decoder_out = torch.cat([model_seen_output, model_masked_output], axis=1)
    
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
        fake_model,
        device,
        fake_signal,
        decoder_out,
        padding_mask,
        encoder_mask,
        decoder_mask,
        decoder_padding_mask
    )
    
    # Make sure constructed signals match,
    assert (full_recon_signal == full_target_signal).all()
    assert (full_recon_signal == simple_fake_signal).all()
    assert (unseen_target_signal == unseen_recon_signal).all()
    # Index into the decoder output signals for unseen signals.
    assert (unseen_target_signal == simple_fake_signal[:, decoder_mask[:GRID_HEIGHT * GRID_WIDTH], :, :]).all()
    # Check patch outputs.
    assert (seen_output == model_seen_output).all()
    assert (seen_target == model_seen_output).all()
    assert (unseen_output == model_masked_output).all()
    assert (unseen_target == model_masked_output).all()
    

def test_get_signal_correlations():
    signal_a = torch.ones(16, 64, 5, 40)
    
    for i in range(signal_a.shape[-1]):
        signal_a[:, :, :, i] *= -1**i * i
    
    signal_b = signal_a * -1
    
    corr_matrix = get_signal_correlations(signal_a, signal_b)
    
    assert corr_matrix.detach().numpy().shape == (64, 5)
    assert torch.isclose(corr_matrix, -torch.ones_like(corr_matrix)).all()
    