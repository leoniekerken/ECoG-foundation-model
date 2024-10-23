import pytest
import torch

import constants
from models import SimpleViT
from mask import get_padding_mask, get_tube_mask, get_decoder_mask

EMBEDDING_DIM = 64
FRAMES_PER_SAMPLE = 40
NUM_BANDS = 5
FRAME_PATCH_SIZE = 4

@pytest.fixture
def model():
    return SimpleViT(
        image_size=[1, constants.GRID_HEIGHT, constants.GRID_WIDTH],
        image_patch_size=[1, 1, 1],
        frames=FRAMES_PER_SAMPLE,
        frame_patch_size=FRAME_PATCH_SIZE,
        dim=EMBEDDING_DIM,
        depth=6,
        heads=6,
        mlp_dim=EMBEDDING_DIM,
        channels=NUM_BANDS,
    )

def test_encoder_forward_without_mask_succeeds(model):
    fake_batch = torch.randn(16, NUM_BANDS, FRAMES_PER_SAMPLE, 1, constants.GRID_HEIGHT, constants.GRID_WIDTH)
    encoder_out = model(fake_batch)
    
    num_patches = FRAMES_PER_SAMPLE * constants.GRID_HEIGHT * constants.GRID_WIDTH // FRAME_PATCH_SIZE
    assert encoder_out.detach().numpy().shape == (16, num_patches, EMBEDDING_DIM)
    

def test_encoder_forward_with_mask_succeeds(model):
    fake_batch = torch.randn(16, NUM_BANDS, FRAMES_PER_SAMPLE, 1, constants.GRID_HEIGHT, constants.GRID_WIDTH)
    # Set two electrodes to nan.
    fake_batch[:, :, :, :, 0, 0] = torch.nan
    fake_batch[:, :, :, :, 0, 1] = torch.nan
    
    padding_mask = get_padding_mask(fake_batch, model, "cpu")
    tube_mask = get_tube_mask(0.5, constants.GRID_HEIGHT, constants.GRID_WIDTH, padding_mask, "cpu")
    
    fake_batch = torch.nan_to_num(fake_batch)
    
    encoder_out = model(fake_batch, encoder_mask=tube_mask, tube_padding_mask=padding_mask)
    
    num_patches = FRAMES_PER_SAMPLE * constants.GRID_HEIGHT * constants.GRID_WIDTH // FRAME_PATCH_SIZE
    num_patches_per_channel = FRAMES_PER_SAMPLE // FRAME_PATCH_SIZE
    assert encoder_out.detach().numpy().shape == (16, (num_patches - 2 * num_patches_per_channel) // 2, EMBEDDING_DIM)
    
    
def test_decoder_forward_with_no_decoder_masking_succeeds(model):
    fake_batch = torch.randn(16, NUM_BANDS, FRAMES_PER_SAMPLE, 1, constants.GRID_HEIGHT, constants.GRID_WIDTH)
    # Set two electrodes to nan.
    fake_batch[:, :, :, :, 0, 0] = torch.nan
    fake_batch[:, :, :, :, 0, 1] = torch.nan
    
    padding_mask = get_padding_mask(fake_batch, model, "cpu")
    tube_mask = get_tube_mask(0.5, constants.GRID_HEIGHT, constants.GRID_WIDTH, padding_mask, "cpu")
    
    fake_batch = torch.nan_to_num(fake_batch)
    
    encoder_out = model(fake_batch, encoder_mask=tube_mask, tube_padding_mask=padding_mask)
    
    decoder_mask = get_decoder_mask(0., tube_mask, "cpu")
    decoder_padding_mask = padding_mask[decoder_mask]
    
    decoder_out = model(encoder_out, encoder_mask=tube_mask, decoder_mask=decoder_mask,
        decoder_padding_mask=decoder_padding_mask)
    
    num_patches = FRAMES_PER_SAMPLE * constants.GRID_HEIGHT * constants.GRID_WIDTH // FRAME_PATCH_SIZE
    num_patches_per_channel = FRAMES_PER_SAMPLE // FRAME_PATCH_SIZE
    assert decoder_out.detach().numpy().shape == (16, num_patches - 2 * num_patches_per_channel, 20)
    
    
def test_decoder_forward_with_decoder_masking_succeeds(model):
    fake_batch = torch.randn(16, NUM_BANDS, FRAMES_PER_SAMPLE, 1, constants.GRID_HEIGHT, constants.GRID_WIDTH)
    # Set two electrodes to nan.
    fake_batch[:, :, :, :, 0, 0] = torch.nan
    fake_batch[:, :, :, :, 0, 1] = torch.nan
    
    padding_mask = get_padding_mask(fake_batch, model, "cpu")
    tube_mask = get_tube_mask(0.5, constants.GRID_HEIGHT, constants.GRID_WIDTH, padding_mask, "cpu")
    
    fake_batch = torch.nan_to_num(fake_batch)
    
    encoder_out = model(fake_batch, encoder_mask=tube_mask, tube_padding_mask=padding_mask)
    
    decoder_mask = get_decoder_mask(0.5, tube_mask, "cpu")
    decoder_padding_mask = padding_mask[decoder_mask]
    
    decoder_out = model(encoder_out, encoder_mask=tube_mask, decoder_mask=decoder_mask,
        decoder_padding_mask=decoder_padding_mask)
    
    num_patches = FRAMES_PER_SAMPLE * constants.GRID_HEIGHT * constants.GRID_WIDTH // FRAME_PATCH_SIZE
    num_patches_per_channel = FRAMES_PER_SAMPLE // FRAME_PATCH_SIZE
    expected_decoder_masks = (num_patches - 2 * num_patches_per_channel) // 2 // 2
    assert decoder_out.detach().numpy().shape == (16, num_patches - 2 * num_patches_per_channel - expected_decoder_masks, 20)
    