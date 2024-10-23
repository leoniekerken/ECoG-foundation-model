import configparser
from dataclasses import dataclass, field, asdict
import json
from argparse import Namespace


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
    # Original sanmpling frequency of data.
    original_fs: int = 512
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
    # Max learning rate for scheduler.
    max_learning_rate: float = 3e-5
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
class LoggingConfig:
    # Directory to write logs to (i.e. tensorboard events, etc).
    event_log_dir: str = "event_logs/"
    # Number of steps to print training progress after.
    print_freq: int = 20

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
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    # Name of training job. Will be used to save metrics.
    job_name: str = None


def create_video_mae_experiment_config_from_file(config_file_path):
    """Convert config file to an experiment config for VideoMAE."""
    # Create fake args which will not error on attribute miss so we can reuse existing function.
    class FakeArgs:
        def __init__(self):
            self.config_file = config_file_path
        
        def __getattr__(self, item):
            return None
        
    return create_video_mae_experiment_config(FakeArgs())


def create_video_mae_experiment_config(args: Namespace | str):
    """Convert command line arguments and config file to an experiment config for VideoMAE.
    
    Config values can be overridden by command line, otherwise use the config file.
    Boolean values can only be overriden to True as of now, to set a flag False do so in the config file.
    
    Can optionally pass 
    """
    config = configparser.ConfigParser(converters={'list': json.loads})
    config.read(args.config_file)
    
    return VideoMAEExperimentConfig(
        video_mae_task_config=VideoMAETaskConfig(
            vit_config=ViTConfig(
                dim=args.dim if args.dim else config.getint("VideoMAETaskConfig.ViTConfig", "dim"),
                mlp_dim=args.mlp_dim if args.mlp_dim else config.getint("VideoMAETaskConfig.ViTConfig", "mlp_dim"),
                patch_size=args.patch_size if args.patch_size else config.getint("VideoMAETaskConfig.ViTConfig", "patch_size"),
                patch_dims=args.patch_dims if args.patch_dims else config.getlist("VideoMAETaskConfig.ViTConfig", "patch_dims"),
                frame_patch_size=args.frame_patch_size if args.frame_patch_size else config.getint("VideoMAETaskConfig.ViTConfig", "frame_patch_size"),
                use_cls_token=args.use_cls_token if args.use_cls_token else config.getboolean("VideoMAETaskConfig.ViTConfig", "use_cls_token"),
            ),
            tube_mask_ratio=args.tube_mask_ratio if args.tube_mask_ratio else config.getfloat("VideoMAETaskConfig", "tube_mask_ratio"),
            decoder_mask_ratio=args.decoder_mask_ratio if args.decoder_mask_ratio else config.getfloat("VideoMAETaskConfig", "decoder_mask_ratio"),
            use_contrastive_loss=args.use_contrastive_loss if args.use_contrastive_loss else config.getboolean("VideoMAETaskConfig", "use_contrastive_loss"),
            running_cell_masking=args.running_cell_masking if args.running_cell_masking else config.getboolean("VideoMAETaskConfig", "running_cell_masking"),
        ),
        trainer_config=TrainerConfig(
            max_learning_rate=args.max_learning_rate if args.max_learning_rate else config.getfloat("TrainerConfig", "max_learning_rate"),
            num_epochs=args.num_epochs if args.num_epochs else config.getint("TrainerConfig", "num_epochs"),
            loss=args.loss if args.loss else config.get("TrainerConfig", "loss"),
        ),
        ecog_data_config=ECoGDataConfig(
            norm=args.norm if args.norm else config.get("ECoGDataConfig", "norm"),
            batch_size=args.batch_size if args.batch_size else config.getint("ECoGDataConfig", "batch_size"),
            data_size=args.data_size if args.data_size else config.getfloat("ECoGDataConfig", "data_size"),
            env=args.env if args.env else config.getboolean("ECoGDataConfig", "env"),
            bands=args.bands if args.bands else config.getlist("ECoGDataConfig", "bands"),
            original_fs = args.original_fs if args.original_fs else config.getint("ECoGDataConfig", "original_fs"),
            new_fs=args.new_fs if args.new_fs else config.getint("ECoGDataConfig", "new_fs"),
            dataset_path=args.dataset_path if args.dataset_path else config.get("ECoGDataConfig", "dataset_path"),
            train_data_proportion=args.train_data_proportion if args.train_data_proportion else config.getfloat("ECoGDataConfig", "train_data_proportion"),
            sample_length=args.sample_length if args.sample_length else config.getint("ECoGDataConfig", "sample_length"),
            shuffle=args.shuffle if args.shuffle else config.getboolean("ECoGDataConfig", "shuffle"),
            test_loader=args.test_loader if args.test_loader else config.getboolean("ECoGDataConfig", "test_loader"),
        ),
        logging_config=LoggingConfig(
            event_log_dir=args.event_log_dir if args.event_log_dir else config.get("LoggingConfig", "event_log_dir"),
            print_freq=args.print_freq if args.print_freq else config.getint("LoggingConfig", "print_freq"),
        ),
        job_name=args.job_name if args.job_name else config.get("JobDetails", "job_name", fallback="train-job"),
    )


def write_config_file(path: str, experiment_config: VideoMAEExperimentConfig):
    """Writes config to path as a .ini file.
    
    Args:
        path (str): path to write file to.
        experiment_config (VideoMAEExperimentConfig): Config to write in .ini format.
    """
    config = configparser.ConfigParser()

    def add_section(section_name, data):
        config[section_name] = {}
        for key, value in data.items():
            config[section_name][key] = str(value)

    add_section("VideoMAETaskConfig.ViTConfig", asdict(experiment_config.video_mae_task_config.vit_config))
    video_mae_task_config = {k: v for k, v in asdict(experiment_config.video_mae_task_config).items() if k != 'vit_config'}
    add_section("VideoMAETaskConfig", video_mae_task_config)
    add_section("ECoGDataConfig", asdict(experiment_config.ecog_data_config))
    add_section("LoggingConfig", asdict(experiment_config.logging_config))
    add_section("TrainerConfig", asdict(experiment_config.trainer_config))
    config["JobDetails"] = {"job_name": experiment_config.job_name}

    # Write the configuration to the file
    with open(path, 'w') as configfile:
        config.write(configfile)