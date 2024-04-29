import argparse
import os
from bisect import bisect_right

import numpy as np
import pandas as pd
from numpy import quantile
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.optim.lr_scheduler import SequentialLR

DEFAULT_CONFIG = {
    "config": None,
    "gpus": 1,
    "tasks": ["CCS", "iRT"],
    "mode": "supervised",
    "pretrained_model": "none",
    "checkpoint_path": None,
    "data_file": None,
    "train_i": None,
    "val_i": None,
    "test_i": None,
    "train_file": None,
    "val_file": None,
    "test_file": None,
    "vocab_file": None,
    "scalers_file": None,
    "lr": 0.001,
    "bs": 1024,
    "accumulate_batches": 1,
    "optim": "SGD",
    "loss": "mae",
    "clip_gradients": True,
    "activation": "gelu",
    "hidden_size": 768,
    "num_layers": 12,
    "seq_len": 50,
    "scheduler": "none",
    "warmup_epochs": 10,
    "cos_freq_epochs": 5,
    "decay_epochs": 50,
}


def d95(y_true, y_pred):
    return quantile(abs(y_true - y_pred), 0.95)


def r(y_true, y_pred):
    return y_true.corr(y_pred)


class Result:
    def __init__(self, y_pred, y_true, run=None):
        assert len(y_true) == len(y_pred)

        self.run = run
        self.y_pred = y_pred
        self.y_true = y_true
        self._e = None
        self._ae = None
        self._mae = None
        self._mse = None
        self._d95 = None
        self._r = None

    def __len__(self):
        return len(self.y_pred)

    def __dict__(self):
        return {
            "run": self.run,
            "y_pred": self.y_pred.values,
            "y_true": self.y_true.values,
            "e": self.e(),
            "ae": self.ae(),
            "mae": self.mae(),
            "mse": self.mse(),
            "d95": self.d95(),
            "r": self.r(),
        }

    def __str__(self):
        return f"Result({str(self.__dict__())})"

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        if item in ["e", "ae", "mae", "mse", "d95", "r"]:
            return getattr(self, item)()
        elif item in ["y_pred", "y_true"]:
            return getattr(self, item).values
        elif item == "run":
            return self.run
        else:
            raise KeyError(item)

    def e(self):
        if self._e is not None:
            return self._e

        self._e = (self.y_pred - self.y_true).values
        return self._e

    def ae(self):
        if self._ae is not None:
            return self._ae

        self._ae = abs(self.e())
        return self._ae

    def mae(self):
        if self._mae is not None:
            return self._mae

        self._mae = np.mean(self.ae())
        return self._mae

    def mse(self):
        if self._mse is not None:
            return self._mse

        self._mse = mean_squared_error(self.y_pred, self.y_true)
        return self._mse

    def d95(self):
        if self._d95 is not None:
            return self._d95

        self._d95 = d95(self.y_pred, self.y_true)
        return self._d95

    def r(self):
        if self._r is not None:
            return self._r

        self._r = r(self.y_pred, self.y_true)
        return self._r


def _get_resized_embeddings(model, old_embeddings, new_num_tokens):
    """Build a resized Embedding Module from a provided token Embedding Module.
        Increasing the size will add newly initialized vectors at the end
        Reducing the size will remove vectors from the end

    Args:
        model:
            Model that needs to be updated (in this function only used for its config)
        old_embeddings:
            The old embeddings of the model, (part of) its weights are kept for the new model
        new_num_tokens: int:
            New number of tokens in the embedding matrix.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

    Return: ``torch.nn.Embeddings``
        Pointer to the resized Embedding Module or the old Embedding Module if
        new_num_tokens is equal to number of tokens in the old embedding
    """
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    if old_num_tokens == new_num_tokens:
        return old_embeddings

    # Build new embeddings
    new_embeddings = nn.Embedding(
        new_num_tokens,
        old_embedding_dim,
        padding_idx=old_embeddings.padding_idx,
    )
    new_embeddings.to(old_embeddings.weight.device)

    # Manually init the new embedding weights
    new_embeddings.weight.data.normal_(
        mean=0.0, std=model.config.initializer_range
    )

    # Copy word embeddings from the previous weights
    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
    new_embeddings.weight.data[:num_tokens_to_copy, :] = (
        old_embeddings.weight.data[:num_tokens_to_copy, :]
    )

    return new_embeddings


def resize_token_embeddings(model, new_num_tokens):
    """Resize input token embeddings matrix of the model if
        new_num_tokens != current embedding size. Take care of tying weights embeddings
        afterwards if the model class has a `tie_weights()` method.

    Arguments:
        model: ProteinModel:
            Model to update with the new embedding
        new_num_tokens: int:
            New number of tokens in the embedding matrix. Increasing the size will add
            newly initialized vectors at the end. Reducing the size will remove vectors
            from the end.
    """
    base_model = getattr(
        model, model.base_model_prefix, model
    )  # get the base model if needed

    old_embeddings = base_model.embeddings.word_embeddings
    new_embeddings = _get_resized_embeddings(
        base_model, old_embeddings, new_num_tokens
    )
    base_model.embeddings.word_embeddings = new_embeddings

    # Update base model and current model config
    model.config.vocab_size = new_num_tokens
    model.vocab_size = new_num_tokens

    # Tie weights again if needed
    if hasattr(model, "tie_weights"):
        model.tie_weights()


class SequentialLRFix(SequentialLR):
    def __init__(
        self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False
    ):
        super().__init__(
            optimizer, schedulers, milestones, last_epoch, verbose
        )

    def step(self, epoch=None):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            self._schedulers[idx].step()
        else:
            self._schedulers[idx].step()
        self._last_lr = self._schedulers[idx].get_last_lr()


SequentialLR = SequentialLRFix


def check_checkpoint_path(d):
    if os.path.isdir(d):
        possible_checkpoints = [f for f in os.listdir(d) if f[-5:] == ".ckpt"]
        if len(possible_checkpoints) == 0:
            raise argparse.ArgumentError(
                argument=None,
                message=f"No checkpoint files found in directory {d}",
            )
        val_losses = []
        for f in possible_checkpoints:
            splitted = [e.split("=") for e in f[:-5].split("-")]
            val_loss = [s[1] for s in splitted if s[0] == "val_loss"][0]
            val_losses.append(val_loss)
        best_ckpt = possible_checkpoints[val_losses.index(min(val_losses))]
        return os.path.join(d, best_ckpt)
    elif os.path.isfile(d) and d[-5:] == ".ckpt":
        return d
    else:
        raise argparse.ArgumentError(
            argument=None,
            message=f"Checkpoint path {d} is nor a directory, "
            f"nor a checkpoint file",
        )


def check_data_files(args):
    if args.data_file is not None:
        if any(
            f is not None
            for f in (args.train_file, args.val_file, args.test_file)
        ):
            raise argparse.ArgumentError(
                argument=None,
                message=f"Ambiguous data arguments: both --data-file and "
                f"--train-file, --val-file, or --test-file given",
            )
        if all(f is None for f in (args.train_i, args.val_i, args.test_i)):
            raise argparse.ArgumentError(
                argument=None,
                message=f"Ambiguous data arguments: --data-file given but no "
                f"train, val or test index file",
            )
        return True

    elif all(f is None for f in (args.train_i, args.val_i, args.test_i)):
        raise argparse.ArgumentError(
            argument=None,
            message=f"Ambiguous data arguments: no --data-file, --train-file, "
            f"--val-file, or --test-file given",
        )
    return False


def split_run_config(path):
    dirs = path.split("/")
    for d in dirs:
        if not d.startswith("CONFIG="):
            continue
        settings = d.split(",")
        return {
            setting.split("=")[0]: setting.split("=")[1]
            for setting in settings
        }
    return None


def end_padding(seq, length, pad_token):
    pad_after = length - len(seq)
    if isinstance(seq, str):
        return seq + pad_token * pad_after
    elif isinstance(seq, list):
        return seq + [pad_token] * pad_after
    else:
        raise TypeError(
            f"seq argument should be a string or list, {type(seq)} given"
        )


def convert_tb_data(root_dir):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """
    from tensorflow.python.framework.errors_impl import DataLossError
    from tensorflow.python.summary.summary_iterator import summary_iterator

    def convert_tfevent(filepath):
        return pd.DataFrame(
            [
                parse_tfevent(
                    e, "/".join(os.path.split(filepath)[0].split("/")[1:])
                )
                for e in summary_iterator(filepath)
                if len(e.summary.value)
            ]
        )

    def parse_tfevent(tfevent, run_name):
        return dict(
            run=run_name,
            wall_time=tfevent.wall_time,
            tag=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )

    columns_order = ["run", "wall_time", "tag", "step", "value"]

    out = []
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            try:
                out.append(convert_tfevent(file_full_path))
            except DataLossError as e:
                print(e)
                print(file_full_path)

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]

    return all_df.reset_index(drop=True)


def get_log_df(filterf=None, cached=True):
    if not os.path.isfile("cache/log_df.csv") or not cached:
        df = convert_tb_data("lightning_logs/")
        os.makedirs("cache", exist_ok=True)
        df.to_csv("cache/log_df.csv")
    else:
        df = pd.read_csv("cache/log_df.csv")

    if filterf is not None:
        runs = filterf(df["run"].unique())
        df = df[df["run"].isin(runs)]

    return df


def config_filter(config, exact=True):
    def skip_run_setting(r, s, t):
        if exact:
            if f"{t}={s}," in r:
                return False
        elif f"{t}={s}" in r:
            return False
        return True

    def f(runs):
        rs = []
        for r in runs:
            if skip_run_setting(r, config, "CONFIG"):
                continue
            rs.append(r)
        return rs

    return f


def get_best_loss_per_run(log_df, loss_column="val_loss"):
    """
    Given a log dataframe (possibly filtered on config before) get the best loss per run, skips runs without any
    values for the loss column

    Currently only the value of the loss column is returned, things like epoch and step can be added
    Currently only 1 loss column is taken into account, since the groupby can be an expensive operation,
    adding the option for multiple loss columns might make sense.

    :param log_df: Dataframe read by get_log_df(...)
    :param loss_column: Name of the loss column
    :return: A dictionary with the best loss per run
    """
    results = {}
    for run, grp in log_df.groupby("run"):
        loss_df = grp[grp["tag"] == loss_column]
        if len(loss_df) == 0:
            print(f"No {loss_column} found for run {run}")
            continue
        best_val = loss_df.loc[loss_df["value"].idxmin()]
        results[run] = best_val["value"]
    return results


def read_predictions(predict_dir, predict_name="predict"):
    for f in os.listdir(predict_dir):
        if f.endswith(".csv") and predict_name in f:
            return pd.read_csv(
                os.path.join(predict_dir, f),
                index_col=0,
                dtype={"Charge": "float64", "DCCS_sequence": str},
            )
    raise FileNotFoundError(
        f"No csv file containing '{predict_name}' found in {predict_dir}"
    )


def calculate_metrics(predict_dir, predict_name="predict", tasks=None):
    if tasks is None:
        tasks = ["iRT", "CCS"]
    df = read_predictions(predict_dir, predict_name)
    task_result = {}
    for t in tasks:
        df_task = df[df["task"] == t]
        task_result[t] = Result(
            df_task["predictions"], df_task["label"], run=predict_dir
        )
    return task_result
