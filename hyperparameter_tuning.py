import argparse
import os.path
import pickle
from argparse import Namespace

import pandas as pd
from ConfigSpace import (
    ConfigurationSpace,
    Float,
    Categorical,
    Integer,
)
from smac import Scenario, HyperparameterOptimizationFacade
from smac.runhistory import TrialValue

from src.util import (
    config_filter,
    get_log_df,
    get_best_loss_per_run,
    split_run_config,
    DEFAULT_CONFIG,
    check_checkpoint_path,
)
from train import post_process_args, train


class MTLParams:
    @property
    def configspace(self):
        cs = ConfigurationSpace(seed=0)
        lr = Float("lr", (0.000001, 0.1), default=0.001, log=True)
        optim = Categorical("optim", ["SGD", "adam", "adamw"], default="adam")
        hidden_size = Integer("hidden_size", (24, 768), q=12, log=True)
        num_layers = Integer("num_layers", (2, 16))
        scheduler = Categorical(
            "scheduler", ["none", "warmup", "warmup_decay_cos"]
        )
        cs.add_hyperparameters([lr, optim, hidden_size, num_layers, scheduler])
        return cs


class MTLParamsNoSize:
    @property
    def configspace(self):
        cs = ConfigurationSpace(seed=0)
        lr = Float("lr", (0.000001, 0.1), default=0.001, log=True)
        optim = Categorical("optim", ["SGD", "adam", "adamw"], default="adam")
        scheduler = Categorical(
            "scheduler", ["none", "warmup", "warmup_decay_cos"]
        )
        cs.add_hyperparameters([lr, optim, scheduler])
        return cs


def read_trial_results(config, cached=True):
    log_df = get_log_df(config_filter(config), cached=cached)

    run_losses = get_best_loss_per_run(log_df)

    trial_loss = {}
    for run, loss in run_losses.items():
        full_config = split_run_config(run)
        trial_loss[int(full_config["TRIALID"])] = loss
    return trial_loss


def update_or_create_configs_and_trials(config, new_configs, old_trials):
    new_config_df = pd.DataFrame(c.config for c in new_configs)
    new_config_df["seed"] = [c.seed for c in new_configs]
    print("Following configs are suggested next:")
    print(new_config_df)

    duplicated_index = []
    if os.path.isfile(f"hpt/{config}.csv"):
        old_config_df = pd.read_csv(f"hpt/{config}.csv", index_col=False)

        new_config_df = pd.concat([old_config_df, new_config_df]).reset_index(
            drop=True
        )
        duplicated_index = new_config_df[
            new_config_df.duplicated()
        ].index.to_list()
        new_config_df = new_config_df.drop_duplicates().reset_index(drop=True)

    os.makedirs("hpt", exist_ok=True)
    new_config_df.to_csv(f"hpt/{config}.csv", index=False)
    trials = old_trials + new_configs

    # Remove duplicated trials (same as in .csv file)
    trials = [t for i, t in enumerate(trials) if i not in duplicated_index]
    pickle.dump(trials, open(f"hpt/{config}.p", "wb"))


def create_configurations(args, cached=True):
    if args.pretrained_model in ["tape", "own"] or args.mode == "pretrain":
        model = MTLParamsNoSize()
    else:
        model = MTLParams()
    scenario = Scenario(model.configspace)
    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario, max_config_calls=1
    )
    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario, n_configs=args.initial_n
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        lambda seed: None,
        intensifier=intensifier,
        initial_design=initial_design,
        overwrite=True,
    )

    trials = []
    if os.path.isfile(f"hpt/{args.config}.p"):
        trials = pickle.load(open(f"hpt/{args.config}.p", "rb"))
        trial_loss_dict = read_trial_results(args.config, cached=cached)
        print(f"Found results for {len(trial_loss_dict)} previous trials")
        for trial_id, loss in trial_loss_dict.items():
            smac.tell(trials[trial_id], TrialValue(loss))
        smac.intensifier._retries = len(trial_loss_dict) + 1

    new_configs = [smac.ask() for _ in range(args.ask_n)]

    update_or_create_configs_and_trials(args.config, new_configs, trials)


def train_from_config(**kwargs):
    config = kwargs["config"]
    i = kwargs["i"]
    config_dict = (
        pd.read_csv(f"hpt/{config}.csv", index_col=False).iloc[i].to_dict()
    )

    config_dict = DEFAULT_CONFIG | kwargs | config_dict | {"config": config}
    run_config = Namespace(**config_dict)
    run_config = post_process_args(run_config)
    run_config.name += f",TRIALID={i}"
    train(run_config)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning configuration or create new configurations"
    )
    parser.add_argument(
        "--hpt-mode", type=str, default="train", choices=["create", "train"]
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-i", type=int, default=None)
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="supervised",
        choices=["supervised", "pretrain"],
    )
    parser.add_argument(
        "-p",
        "--pretrained-model",
        default="none",
        type=str,
        choices=["none", "tape", "own"],
    )
    parser.add_argument("--data-file", type=str)
    parser.add_argument("--train-i", type=str)
    parser.add_argument("--val-i", type=str)
    parser.add_argument("--test-i", type=str)
    parser.add_argument("--vocab-file", type=str)
    parser.add_argument("--scalers-file", type=str)
    parser.add_argument("--bs", default=1024, type=int)
    parser.add_argument("-a", "--accumulate-batches", default=1, type=int)
    parser.add_argument("--hidden-size", default=768, type=int)
    parser.add_argument("--num-layers", default=12, type=int)
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
    parser.add_argument(
        "--initial-n",
        type=int,
        default=25,
        help="Number of initial configurations to generate. Set to 0 when you already generated initial configs",
    )
    parser.add_argument(
        "--ask-n",
        type=int,
        default=25,
        help="Number of configurations to generate.",
    )

    args = parser.parse_args()

    if args.hpt_mode == "train" and args.i is None:
        raise ValueError("Argument -i is required when --hpt-mode is train")

    args.hpt_config = None

    return args


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.expand_frame_repr", False)  # Don't wrap columns
    pd.set_option("display.max_colwidth", 100)

    args = parse_args()

    if args.hpt_mode == "create":
        create_configurations(args, cached=False)
    elif args.hpt_mode == "train":
        train_from_config(**args.__dict__)
