from einops.layers.torch import Rearrange
import torch

from mask import get_padding_mask, get_tube_mask, get_decoder_mask


def test_get_padding_mask():
    patch_depth = 1
    patch_height = 1
    patch_width = 1
    frame_patch_size = 4
    class FakeModel():
        def __init__(self):
            self.patchify = Rearrange(
                "b c (f pf) (d pd) (h ph) (w pw) -> b f d h w (pd ph pw pf c)",
                pd=patch_depth,
                ph=patch_height,
                pw=patch_width,
                pf=frame_patch_size,
            )
        
    model = FakeModel()
    # [batch, channels, frames, depth, height, width]
    fake_signal = torch.ones((4, 5, 8, 1, 2, 2))
    # Set top row of electrodes to nan implying padding
    fake_signal[:, :, :, :, 0, 0] = torch.nan
    fake_signal[:, :, :, :, 0, 1] = torch.nan
    
    actual_padding_mask = get_padding_mask(fake_signal, model, "cpu")
    
    assert torch.all(actual_padding_mask == torch.tensor([False, False, True, True, False, False, True, True]))
    
    
def test_get_tube_mask():
    padding_mask = torch.tensor([False, False, True, True, False, False, True, True])
    
    tube_mask = get_tube_mask(0.5, 2, 2, padding_mask, "cpu")
    
    # tube mask is random but make sure 50% of patches are masked and it is using tube masking.
    assert 2 * tube_mask.sum() == padding_mask.sum()
    assert tube_mask[:4].sum() == 1
    
    
def test_get_decoder_mask_always_flips_sign_for_masked_values():
    tube_mask = torch.tensor([False, False, False, False, False, False, True, True])
    
    decoder_mask = get_decoder_mask(0., tube_mask, "cpu")
    
    assert (decoder_mask == torch.tensor([True, True, True, True, True, True, False, False])).all()
    

def test_get_decoder_mask_can_mask_values():
    tube_mask = torch.tensor([False, False, False, False, False, False, True, True])
    
    decoder_mask = get_decoder_mask(0.5, tube_mask, "cpu")
    
    # Random so can't guarantee exact value setting.
    assert decoder_mask.sum() == 3