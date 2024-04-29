import os
import pickle
from argparse import Namespace

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from tape.models.modeling_bert import ProteinBertConfig
from torch.utils.data import DataLoader

from src.dataset import MTLPepDataset, custom_collate
from src.lit_model import LitMTL
from src.read_data import apply_index_file
from src.util import (
    DEFAULT_CONFIG,
    check_checkpoint_path,
    split_run_config,
)


def predict(run, args, run_config):
    args.vocab = pickle.load(open(args.vocab_file, "rb"))
    args.scalers = pickle.load(open(args.scalers_file, "rb"))

    all_data_df = pd.read_csv(args.all_data_file, index_col=0)
    args.df_test = apply_index_file(all_data_df, args.predict_i)

    predict_ds = MTLPepDataset(args.df_test, args)
    predict_dl = DataLoader(
        predict_ds,
        batch_size=args.bs,
        collate_fn=custom_collate,
        num_workers=1,
    )

    bert_config = ProteinBertConfig.from_pretrained(
        "bert-base",
        vocab_size=len(args.vocab),
        hidden_act=run_config["ACTIVATION"],
        hidden_size=int(run_config["SIZE"]),
        intermediate_size=int(run_config["SIZE"]) * 4,
        num_hidden_layers=int(run_config["NUMLAYERS"]),
    )

    lit_model = LitMTL.load_from_checkpoint(
        check_checkpoint_path(os.path.join(run, "checkpoints")),
        mtl_config=args,
        bert_config=bert_config,
    )

    name, version = run.split("/")[-2:]
    logger = TensorBoardLogger("./lightning_logs", name=name, version=version)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        logger=logger,
        strategy=(
            "ddp_find_unused_parameters_true" if args.gpus > 1 else "auto"
        ),
        precision="16-mixed",
    )
    trainer.test(lit_model, dataloaders=predict_dl)


def predict_run(run, all_data_file, predict_i):
    data_config = {
        "all_data_file": all_data_file,
        "predict_i": predict_i,
        "vocab_file": os.path.join(run, "vocab.p"),
        "scalers_file": os.path.join(run, "scalers.p"),
    }

    run_config = split_run_config(run)
    config_dict = DEFAULT_CONFIG | data_config
    args = Namespace(**config_dict)
    args.predict_file_name = "predict"
    predict(run, args, run_config)


if __name__ == "__main__":

    # Example on how to create predictions with an existing model
    best_run = (
        "lightning_logs/CONFIG=mtl_5foldcv_finetune_tape_0,TASKS=CCS_iRT,MODE=supervised,PRETRAIN=tape,"
        "LR=0.0003105497384738,BS=1024,OPTIM=adamw,LOSS=mae,CLIP=False,ACTIVATION=gelu,SCHED=warmup_decay_cos,"
        "SIZE=768,NUMLAYERS=12/version_0"
    )

    predict_run(
        best_run,
        "data/mtl_5fold_cv/all_data.csv",
        "data/mtl_5fold_cv/test_0.csv",
    )
