import argparse
import ast


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="configs/video_mae_train.ini")
    parser.add_argument("--job-name", type=str)
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.set_defaults(debug=False)
    parser.add_argument("--test-loader", dest="test_loader", action="store_true")
    parser.set_defaults(test_loader=False)
    parser.add_argument("--data-size", type=float)
    parser.add_argument("--shuffle", dest="shuffle", action="store_true")
    parser.set_defaults(shuffle=False)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--env", dest="env", action="store_true")
    parser.set_defaults(env=False)
    parser.add_argument("--norm", type=str)
    parser.add_argument("--original-fs", type=int)
    parser.add_argument("--new-fs", type=int)
    parser.add_argument("--sample-length", type=int)
    parser.add_argument("--patch-dims", type=str)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--frame-patch-size", type=int)
    parser.add_argument("--tube-mask-ratio", type=float)
    parser.add_argument("--decoder-mask-ratio", type=float)
    parser.add_argument(
        "--running-cell-masking", dest="running_cell_masking", action="store_true"
    )
    parser.add_argument("--bands", type=str)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--max-learning-rate", type=float)
    parser.add_argument("--use-cls-token", dest="use_cls_token", action="store_true")
    parser.add_argument(
        "--use-contrastive-loss", dest="use_contrastive_loss", action="store_true"
    )
    parser.add_argument("--dim", type=int)
    parser.add_argument("--mlp-dim", type=int)
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Relative path to the root of the dataset folder.",
    )
    parser.add_argument(
        "--train-data-proportion",
        type=float,
        help="Percentage of data to assign to train split. All remaining data is assigned to test split.",
    )
    parser.add_argument(
        "--event-log-dir",
        type=str,
        help="Directory to write training logs to (i.e. tensorboard logs)",
    )
    parser.add_argument(
        "--print-freq",
        type=int,
        help="Number of steps to print training progress after.",
    )
    args = parser.parse_args()

    # parse string input to list of lists
    args.bands = ast.literal_eval(args.bands) if args.bands else None
    args.patch_dims = ast.literal_eval(args.patch_dims) if args.patch_dims else None

    return args
