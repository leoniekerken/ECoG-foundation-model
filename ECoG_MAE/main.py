import logging
import sys
from parser import arg_parser
from ecog_setup import system_setup, model_setup
from config import create_video_mae_experiment_config
from loader import dl_setup
from models import *
from tests import test_loader, test_model
from train import train_model

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def main(args):
    
    experiment_config = create_video_mae_experiment_config(args)

    accelerator, device, data_type, local_rank = system_setup()
    train_dl, test_dl, num_train_samples = dl_setup(experiment_config)
    model, optimizer, lr_scheduler, _ = model_setup(
        experiment_config, device, num_train_samples
    )

    if args.test_loader:
        # TODO: Can migrate test loader to use new configs as well but for now
        test_loader(experiment_config, train_dl, test_dl)
        # test_model(args, device, model, num_patches)
        sys.exit("Stopping the script")

    model = train_model(
        experiment_config,
        device,
        model,
        train_dl,
        test_dl,
        optimizer,
        lr_scheduler,
        accelerator,
        data_type,
        local_rank,
    )


if __name__ == "__main__":

    args = arg_parser()
    main(args)
