from dataclasses import dataclass, field


# Config classes here are very roughly following the format of Tensorflow Model Garden: https://www.tensorflow.org/guide/model_garden#training_framework
# to try and make expanding to new models and tasks slightly easier by logically breaking up the parameters to training into distinct pieces and directly
# documenting the fields which can be configured.
@dataclass
class ECoGDataConfig:
    # If 'batch' then will normalize data within a batch.
    norm: str = None
    # Percentage of data to include in training/testing.
    data_size: float = 1.0
    # Batch size to train with.
    batch_size: int = 32
    # If true then convert data to power envelope by taking magnitude of Hilbert
    # transform.
    env: bool = False
    # Frequency bands for filtering raw iEEG data.
    bands: list[list[int]] = field(
        default_factory=lambda: [[4, 8], [8, 13], [13, 30], [30, 55], [70, 200]]
    )
    # Frequency to resample data to.
    new_fs: int = 20
    # Relative path to the dataset root directory.
    dataset_path: str = None
    # Proportion of data to have in training set. The rest will go to test set.
    train_data_proportion: float = 0.9
    # Number of seconds of data to use for a training example.
    sample_length: int = 2
    # If true then shuffle the data before splitting to train and eval.
    shuffle: bool = False
    # If True then uses a mock data loader.
    test_loader: bool = False


@dataclass
class TrainerConfig:
    # Learning rate for training. If 0 then uses Adam scheduler.
    learning_rate: float = 0.0
    # Number of epochs to train over data.
    num_epochs: int = 10
    # Type of loss to use.
    loss: str = "patch"


@dataclass
class ViTConfig:
    # Dimensionality of token embeddings.
    dim: int = 512
    # Dimensionality of feedforward network after attention layer.
    mlp_dim: int = 512
    # The number of electrodes in a patch.
    patch_size: int = 0
    # The number of frames to include in a tube per video mae.
    frame_patch_size: int = 0
    # Specifies per dimension patch size. [depth, height, width]
    patch_dims: list[int] = field(default_factory=lambda: [1, 1, 1])
    # Prepend classification token to input if True. Always True if
    # use_contrastive_loss is True.
    use_cls_token: bool = False


@dataclass
class VideoMAETaskConfig:
    # Config for model.
    vit_config: ViTConfig = field(default_factory=ViTConfig)
    # Proportion of tubes to mask out. See VideoMAE paper for details.
    tube_mask_ratio: float = 0.5
    # The ratio of the number of masked tokens in the input sequence.
    decoder_mask_ratio: float = 0
    # If true then use contrastive loss to train model. Currently not supported.
    use_contrastive_loss: bool = False
    # If true use running cell masking when masking tokens.
    running_cell_masking: bool = False


@dataclass
class VideoMAEExperimentConfig:
    video_mae_task_config: VideoMAETaskConfig = field(
        default_factory=VideoMAETaskConfig
    )
    ecog_data_config: ECoGDataConfig = field(default_factory=ECoGDataConfig)
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    # Name of training job. Will be used to save metrics.
    job_name: str = None


def create_video_mae_experiment_config(args):
    """Convert command line arguments to an experiment config for VideoMAE."""
    return VideoMAEExperimentConfig(
        video_mae_task_config=VideoMAETaskConfig(
            vit_config=ViTConfig(
                dim=args.dim,
                mlp_dim=args.mlp_dim,
                patch_size=args.patch_size,
                patch_dims=args.patch_dims,
                frame_patch_size=args.frame_patch_size,
                use_cls_token=args.use_cls_token,
            ),
            tube_mask_ratio=args.tube_mask_ratio,
            decoder_mask_ratio=args.decoder_mask_ratio,
            use_contrastive_loss=args.use_contrastive_loss,
            running_cell_masking=args.running_cell_masking,
        ),
        trainer_config=TrainerConfig(
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            loss=args.loss,
        ),
        ecog_data_config=ECoGDataConfig(
            norm=args.norm,
            batch_size=args.batch_size,
            data_size=args.data_size,
            env=args.env,
            bands=args.bands,
            new_fs=args.new_fs,
            dataset_path=args.dataset_path,
            train_data_proportion=args.train_data_proportion,
            sample_length=args.sample_length,
            shuffle=args.shuffle,
            test_loader=args.test_loader,
        ),
        job_name=args.job_name,
    )
