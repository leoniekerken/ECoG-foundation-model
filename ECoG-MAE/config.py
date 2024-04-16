# TODO implement learning rate scheduler for better performance - initially Paul had implemented one, but we are using a fixed LR for now ...

import os
import torch
from accelerate import Accelerator, DeepSpeedPlugin

import utils
from models import * 

def system_setup():

    """
    Sets up accelerator, device, datatype precision and local rank

    Args:
        
    Returns:
        accelerator: an accelerator instance - https://huggingface.co/docs/accelerate/en/index
        device: the gpu to be used for model training
        data_type: the data type to be used, we use "fp16" mixed precision - https://towardsdatascience.com/understanding-mixed-precision-training-4b246679c7c4
        local_rank: the local rank environment variable (only needed for multi-gpu training)
    """
    
    # tf32 data type is faster than standard float32
    torch.backends.cuda.matmul.allow_tf32 = True

    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    device = "cuda:0"

    # set data_type to match your mixed precision
    if accelerator.mixed_precision == "bf16":
        data_type = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        data_type = torch.float16
    else:
        data_type = torch.float32

    # only need this if we want to set up multi GPU training
    local_rank = os.getenv("RANK")
    if local_rank is None:
        local_rank = 0
    else:
        local_rank = int(local_rank)

    return accelerator, device, data_type, local_rank

def model_setup(args, device):

    """
    Sets up model config

    Args:
        args: input arguments
        device: cuda device
        
    Returns:
        model: an untrained model instance with randomly initialized parameters 
        optimizer: an Adam optimizer instance - https://www.analyticsvidhya.com/blog/2023/12/adam-optimizer/ 
        num_patches: the number of patches in which the input data is segmented
    """

    ### class token config ###
    use_cls_token = args.use_cls_token

    ### Loss Config ###
    use_contrastive_loss = args.use_contrastive_loss
    constrastive_loss_weight = 1.0
    use_cls_token = (
        True if use_contrastive_loss else use_cls_token
    )  # if using contrastive loss, we need to add a class token

    input_size = [1, 8, 8]
    print("input_size", input_size)
    num_frames = args.sample_length * args.new_fs

    img_size = (1, 8, 8)
    patch_size = tuple(args.patch_size)
    frame_patch_size = args.frame_patch_size
    num_patches = int(  # Defining the number of patches
        (img_size[0] / patch_size[0])
        * (img_size[1] / patch_size[1])
        * (img_size[2] / patch_size[2])
        * num_frames
        / frame_patch_size
    )

    num_encoder_patches = int(num_patches * (1 - args.tube_mask_ratio))
    num_decoder_patches = int(num_patches * (1 - args.decoder_mask_ratio))
    print("num_patches", num_patches)
    print("num_encoder_patches", num_encoder_patches)
    print("num_decoder_patches", num_decoder_patches)

    max_lr = 3e-5  # 3e-5 seems to be working best? original videomae used 1.5e-4

    model = SimpleViT(
        image_size=img_size,  # depth, height, width
        image_patch_size=patch_size,  # depth, height, width patch size - change width from patch_size to 1
        frames=num_frames,
        frame_patch_size=frame_patch_size,
        depth=12,
        heads=12,
        dim=512,
        mlp_dim=512,  # TODO: right now dim needs to equal mlp_dim, and both need to be 512
        num_encoder_patches=num_encoder_patches,
        num_decoder_patches=num_decoder_patches,
        channels=len(args.bands),
        use_rope_emb=False,
        use_cls_token=False,
    )
    utils.count_params(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    opt_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-2,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

    # TODO implement lr scheduler

    print("\nDone with model preparations!")

    return model, optimizer, num_patches