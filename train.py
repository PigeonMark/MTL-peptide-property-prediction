import argparse
import pickle
import torch

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.dataset import MTLPepDataset, custom_collate
from src.model_util import EarlyStoppingLate, create_model
from src.read_data import read_train_val_test_data
from src.util import (
    check_checkpoint_path,
    split_run_config,
    check_data_files,
)
from src.vocab import create_vocab


def get_vocab(args):
    if args.vocab_file is None:
        vocab = create_vocab(args)
        if args.use_1_data_file:
            vocab_path = args.data_file[:-4] + "_vocab.p"
        else:
            vocab_path = args.train_file[:-4] + "_vocab.p"
        pickle.dump(vocab, open(vocab_path, "wb"))
    else:
        vocab = pickle.load(open(args.vocab_file, "rb"))
    return vocab


def get_scalers(args, df_train=None):
    if args.scalers_file is None:
        scalers = {}
        for t in args.tasks:
            train_df_t = df_train[df_train["task"] == t]
            t_scaler = StandardScaler()
            if len(train_df_t) > 0:
                t_scaler.fit(train_df_t["label"].values.reshape(-1, 1))
            else:
                # If no training values for a task, create a scaler that does not alter the values
                t_scaler.fit([[-1.0], [1.0]])
            scalers[t] = t_scaler

        if args.use_1_data_file:
            scalers_path = args.train_i[:-4] + "_scalers.p"
        else:
            scalers_path = args.train_file[:-4] + "_scalers.p"
        pickle.dump(scalers, open(scalers_path, "wb"))
    else:
        scalers = pickle.load(open(args.scalers_file, "rb"))
    return scalers


def train(args):
    torch.set_float32_matmul_precision("medium")
    print(f"Training with this configuration:\n{args}")
    logger = TensorBoardLogger("./lightning_logs", name=args.name)
    print(f"Logging to {logger.log_dir}")

    print("Reading train and validation data")
    args.df_train, args.df_val, args.df_test = read_train_val_test_data(args)

    print("Creating or loading the vocab")
    args.vocab = get_vocab(args)

    if args.mode == "supervised":
        print("Creating or loading the scaler")
        args.scalers = get_scalers(args, args.df_train)

    print("Creating the datasets")
    train_ds = MTLPepDataset(args.df_train, args)
    val_ds = MTLPepDataset(args.df_val, args)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=1,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.bs,
        collate_fn=custom_collate,
        num_workers=1,
    )

    lit_model = create_model(args)

    trainer = pl.Trainer(
        max_epochs=-1,
        min_epochs=15,
        accelerator="gpu",
        devices=args.gpus,
        strategy=(
            "ddp_find_unused_parameters_true" if args.gpus > 1 else "auto"
        ),
        accumulate_grad_batches=args.accumulate_batches,
        logger=logger,
        gradient_clip_val=(0.5 if args.clip_gradients else None),
        precision="16-mixed",
        profiler="simple",
        callbacks=[
            EarlyStoppingLate(
                monitor="val_loss",
                patience=25,
                verbose=True,
                mode="min",
            ),
            ModelCheckpoint(
                filename="{epoch}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        ],
    )
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )

    if args.df_test is not None:
        test_ds = MTLPepDataset(args.df_test, args)
        test_dl = DataLoader(
            test_ds,
            batch_size=args.bs,
            collate_fn=custom_collate,
        )
        trainer.test(ckpt_path="best", dataloaders=test_dl)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi Task Learning Transformer for peptide property prediction"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Give a name to the run, can be used to filter different models afterwards "
        '(default: "default")',
    )

    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs to use, default: 1. It is adviced to only use more than 1 for "
        "(pre)training, it can give unexpected behaviour when e.g. saving predictions",
    )

    # Model parameters
    parser.add_argument(
        "-t",
        "--tasks",
        type=str,
        nargs="+",
        default=["CCS", "iRT"],
        choices=["iRT", "CCS"],
        help="Tasks for the multitask model, currently only 'iRT' and 'CCS' are supported. Default: "
        "CCS and iRT",
    )

    # General train parameters
    parser.add_argument(
        "-m",
        "--mode",
        default="supervised",
        type=str,
        choices=["supervised", "pretrain"],
        help="Training mode: 'supervised' (default) for finetuning or training from scratch or "
        "'pretrain' for pretraining with own data",
    )

    parser.add_argument(
        "-p",
        "--pretrained-model",
        default="none",
        type=str,
        choices=["none", "tape", "own"],
        help="Which pretrained model to use: 'none' (default) to train from scratch, 'tape' to use the "
        "pretrained TAPE model and 'own' to use a self-pretrained model",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=check_checkpoint_path,
        help="Path to checkpoint file or directory containing checkpoint file when finetuning a model "
        "pretrained ourselves. In case a directory is given, the best validation checkpoint will be used. "
        "As an alternative --checkpoint-id can be used instead.",
    )
    parser.add_argument(
        "--checkpoint-id",
        type=int,
        default=None,
        help="Index of the checkpoint path to use from config/checkpoints.csv. This is an alternative to "
        "giving the full checkpoint path with '--checkpoint-path'",
    )

    # Data parameters
    parser.add_argument(
        "-d",
        "--data-file",
        type=str,
        default=None,
        help="CSV file that contains all data, use together with train, validation (and test) index "
        "files to get the final train/val/test data. Alternatively use separate train-file, "
        "validation-file (and test-file)",
    )

    parser.add_argument(
        "--train-i",
        type=str,
        default=None,
        help="CSV file containing the indices from the 'data-file' to use "
        "for training",
    )

    parser.add_argument(
        "--val-i",
        type=str,
        default=None,
        help="CSV file containing the indices from the 'data-file' to use "
        "for validation",
    )

    parser.add_argument(
        "--test-i",
        type=str,
        default=None,
        help="CSV file containing the indices from the 'data-file' to use "
        "for testing",
    )

    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="CSV file that contains the training data.",
    )

    parser.add_argument(
        "--val-file",
        type=str,
        default=None,
        help="CSV file that contains the validation data.",
    )

    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="CSV file that contains the test data.",
    )

    parser.add_argument(
        "--vocab-file",
        type=str,
        default=None,
        help="Vocab to use when finetuning or predicting with a model previously trained. Will create "
        "a new vocab from the given data when None. The vocab encodes each amino acid-PTM "
        "combination as a unique token",
    )

    parser.add_argument(
        "--scalers-file",
        type=str,
        default=None,
        help="Scalers to use for label standardization (should be a pickle of dict containing 1 scikit "
        "StandardScaler per task) Will create new scalers from the training data when None.",
    )

    # NN hyperparameters
    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        help="The learning rate (default 0.001)",
    )

    parser.add_argument(
        "--bs",
        default=1024,
        type=int,
        help="The batch size (default 1024)",
    )

    parser.add_argument(
        "-a",
        "--accumulate-batches",
        default=1,
        type=int,
        help="The number of batches to accumulate before updating the weights (default 1), "
        "can be used when a full batch does not fit in (GPU) memory. Using e.g. --bs 8 -a 4 and "
        "--bs 32 -a 1 gives a similar result and uses less memory but more runtime",
    )

    parser.add_argument(
        "-o",
        "--optim",
        default="SGD",
        type=str,
        choices=["SGD", "adamw", "adam"],
        help="The optimizer to use (default SGD)",
    )

    parser.add_argument(
        "-l",
        "--loss",
        default="mae",
        type=str,
        choices=["mae", "mse"],
        help="Loss metric minimized to optimize the model",
    )

    parser.add_argument(
        "-c",
        "--clip-gradients",
        default=False,
        action="store_true",
        help="Clip gradients (default False)",
    )

    parser.add_argument(
        "--activation",
        default="gelu",
        type=str,
        choices=["gelu", "relu"],
        help="Transformer activation function (default gelu)",
    )

    parser.add_argument(
        "--hidden-size",
        default=768,
        type=int,
        help="Hidden size for the Transformer layers (default 768)",
    )

    parser.add_argument(
        "--num-layers",
        default=12,
        type=int,
        help="Number of Transformer layers (default 12)",
    )

    parser.add_argument(
        "--seq-len",
        default=50,
        type=int,
        help="Sequence length (default 50), shorter sequences will "
        "be padded, longer sequences will be truncated.",
    )

    # Schedular parameters
    parser.add_argument(
        "--scheduler",
        default="none",
        choices=["none", "warmup", "warmup_decay_cos"],
        type=str,
        help="Scheduler to use (default none), when warmup_decay_cos is selected a warmup phase + "
        "a CosineAnnealingWarmRestarts scheduler with linear decay is used.",
    )

    parser.add_argument(
        "--warmup-epochs",
        default=10,
        type=int,
        help="Number of epoch for the warmup phase (default 10). Only used when scheduler is "
        "warmup_decay_cos.",
    )

    parser.add_argument(
        "--cos-freq-epochs",
        default=5,
        type=int,
        help="The frequency of the CosineAnnealing function in number of epochs (default 5). "
        " Only used when scheduler is warmup_decay_cos.",
    )
    parser.add_argument(
        "--decay_epochs",
        default=50,
        type=int,
        help="The number of epochs over which to linearly decay the LR from 100%% to 10%% "
        "(default 50). Only used when scheduler is warmup_decay_cos.",
    )
    parser.add_argument(
        "--hpt-config",
        default=None,
        type=str,
        help="Use model arguments specified in a hpt config file. The arguments present in "
        "the hpt file will overwrite the other arguments",
    )
    parser.add_argument(
        "--hpt-id",
        default=None,
        type=int,
        help="The id from the hpt-config file to use",
    )

    args = parser.parse_args()

    args = post_process_args(args)

    return args


def post_process_args(args):
    args.use_1_data_file = check_data_files(args)

    if args.pretrained_model == "own":
        if args.checkpoint_path is None and args.checkpoint_id is None:
            raise ValueError(
                f"--pretrained_model was set to 'own' but no checkpoint path was provided."
            )
        if args.checkpoint_path is None:
            args.checkpoint_path = pd.read_csv(
                "config/checkpoints.csv", index_col=0
            ).iloc[args.checkpoint_id]["checkpoint_path"]
            args.checkpoint_path = check_checkpoint_path(args.checkpoint_path)

        configs = split_run_config(args.checkpoint_path)
        args.hidden_size = int(configs.get("SIZE", 768))
        args.num_layers = int(configs.get("NUMLAYERS", 12))
        print(
            f"Finetuning own model, updated hidden size to {args.hidden_size} and num layers to {args.num_layers}"
        )

    if args.pretrained_model == "tape" and (
        args.hidden_size != 768 or args.num_layers != 12
    ):
        print(
            f"--pretrained_model was set to 'tape', hidden-size will always be 768 and num-layers 12"
        )

    if args.hpt_config is not None:
        assert args.hpt_id is not None
        config_dict = (
            pd.read_csv(args.hpt_config, index_col=False)
            .iloc[args.hpt_id]
            .to_dict()
        )
        if "seed" in config_dict:
            del config_dict["seed"]

        args = argparse.Namespace(**(vars(args) | config_dict))
        print(f"Updated with hpt config: {config_dict}")

    # Sort the tasks so the order will always be the same
    args.tasks = sorted(args.tasks)

    args.name = (
        f"CONFIG={args.config},TASKS={'_'.join(args.tasks)},MODE={args.mode},"
        f"PRETRAIN={args.pretrained_model},LR={args.lr},BS={args.bs * args.accumulate_batches},"
        f"OPTIM={args.optim},LOSS={args.loss},CLIP={args.clip_gradients},ACTIVATION={args.activation},"
        f"SCHED={args.scheduler},SIZE={args.hidden_size},NUMLAYERS={args.num_layers}"
    )
    return args


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.expand_frame_repr", False)  # Don't wrap columns
    pd.set_option("display.max_colwidth", 100)
    commandline_args = parse_args()
    train(commandline_args)
