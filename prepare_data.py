import os
import pickle

import pandas as pd
import yaml


def get_mods():
    """
    Read the configuration file with the PTM information

    :return: a dictionary with the PTM information
    """
    return yaml.safe_load(open("config/modifications.yaml", "r"))


def replace_modifications(df, replace_dict, seq_col):
    """
    Replace all modifications with their UNIMOD notation using a dict
    :param df: dataframe containing the data
    :param replace_dict: dictionary with as keys the PTMs used in the data and as values their unimod ids.
    See config/modifications.yaml for examples
    :param seq_col: The name of the column of the df that contains the modified sequences
    :return: the same dataframe but with replaced PTMs
    """
    for mod, unimod_id in replace_dict.items():
        df[seq_col] = df[seq_col].str.replace(mod, f"[UNIMOD:{unimod_id}]")
    return df


def get_unmodified_seqs(df, mod_seq_col="modified_sequence"):
    """
    Get a series of the sequences without modifications (all sequences will be returned).
    Only works after replacing modifications with their UNIMOD notation!
    :param df: the dataframe containing the data
    :param mod_seq_col: the name of the column of the df that contains the modified sequences
    :return: a Series with the same sequences as the mod_seq_col column but without modifications
    """
    unmod_seqs = df[mod_seq_col].str.replace(
        "\[UNIMOD:[0-9]*\]", "", regex=True
    )
    return unmod_seqs


def merge_duplicates(
    df, mod_seq_col="modified_sequence", label_col="iRT", merge_method="median"
):
    """
    Merge duplicate modified sequences. Looks for duplicates in mod_seq_col and merges the label_col values
    by using the merge_method
    :param df: Dataframe containing the data
    :param mod_seq_col: The name of the column of the df that contains the modified sequences
    :param label_col: The name of the column of the df that contains the labels (e.g. iRT or CCS)
    :param merge_method: Method used to merge the label values of duplicate modified sequences,
    currently only median and mean are implemented
    :return: df with duplicate modified sequences merged
    """
    if merge_method == "median":
        labels = df.groupby(mod_seq_col)[label_col].median()
    elif merge_method == "mean":
        labels = df.groupby(mod_seq_col)[label_col].mean()
    else:
        raise RuntimeError(
            f"Argument merge_method {merge_method} not supported"
        )

    no_dup = df.drop_duplicates(mod_seq_col).drop(columns=[label_col])
    return pd.merge(
        no_dup,
        labels,
        how="outer",
        left_on=mod_seq_col,
        right_index=True,
        validate="1:1",
    ).reset_index(drop=True)


def remove_chron_duplicate_labels(df):
    """
    Chronologer has iRT values that occur for many different peptides. This is not normal and those peptides are removed
    with this function.

    We round the iRT to 10 digits because some of the duplicate values differ on the last digit. We count the number
    of times each iRT value occurs within each Source and remove those that occur more than 300 times.
    """
    rounded_df = df.round({"label": 10})
    counts = (
        rounded_df.value_counts(subset=["Source", "label"])
        .to_frame()
        .reset_index()
    )

    bad_counts = counts[counts["count"] >= 300]
    to_drop = []
    for rt, source in zip(bad_counts["label"], bad_counts["Source"]):
        to_drop.extend(
            rounded_df[
                (rounded_df["Source"] == source) & (rounded_df["label"] == rt)
            ].index
        )
    df = df.drop(to_drop)
    return df


def process_chronologer(file_path, from_cache=True):
    cache_file = os.path.join(
        "cache", f"{os.path.normpath(file_path).replace('/', '_')}.p"
    )
    if from_cache and os.path.isfile(cache_file):
        return pickle.load(open(cache_file, "rb"))

    modifications = get_mods()
    df = pd.read_csv(
        file_path, sep="\t", usecols=["Source", "PeptideModSeq", "Prosit_RT"]
    )
    df = df.rename(
        columns={"PeptideModSeq": "modified_sequence", "Prosit_RT": "label"}
    )
    df = replace_modifications(
        df, modifications["Chronologer"], "modified_sequence"
    )
    df_len = len(df)
    print(f"Got {df_len} peptides from {file_path}")

    # remove all peptides with an iRT over 150, these are not reliable
    df = df[df["label"] < 150]
    print(
        f"Removed {df_len - len(df)} peptides with iRT > 150, {len(df)} remaining"
    )
    df_len = len(df)

    # remove all peptides with an iRT value that occurs 300 or more times
    df = remove_chron_duplicate_labels(df)
    del df["Source"]
    print(
        f"Removed {df_len - len(df)} peptides with duplicate iRT values, {len(df)} remaining"
    )
    df_len = len(df)

    # Remove samples with impossible modifications
    df = df[~df["modified_sequence"].str.contains("M[UNIMOD:1]", regex=False)]
    df = df[~df["modified_sequence"].str.contains("Q[UNIMOD:1]", regex=False)]

    print(
        f"Removed {df_len - len(df)} peptides with impossible PTMs, {len(df)} remaining"
    )
    df_len = len(df)

    # Merge duplicate peptides by taking the median iRT
    df = merge_duplicates(df, label_col="label")

    print(
        f"Removed {df_len - len(df)} duplicate peptides, {len(df)} remaining (final)"
    )

    df["task"] = "iRT"

    pickle.dump(df, open(cache_file, "wb"))

    return df


def process_DCCS(file_paths, from_cache=True):
    cache_file = os.path.join("cache", f"DCCS.p")
    if from_cache and os.path.isfile(cache_file):
        return pickle.load(open(cache_file, "rb"))

    dfs = []
    for f in file_paths:
        df = pd.read_csv(f, usecols=["Modified_sequence", "CCS", "Charge"])
        print(f"{f} has {len(df)} peptides")
        df_len = len(df)
        df = df.dropna().rename(
            columns={"Modified_sequence": "modified_sequence", "CCS": "label"}
        )
        print(
            f"Removed {df_len - len(df)} peptides with missing values from {f}"
        )
        dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)
    print(f"Merged, {len(df)} peptides")
    df["DCCS_sequence"] = df["modified_sequence"]
    print(df["Charge"].value_counts() * 100 / len(df))

    modifications = get_mods()
    # Convert modification to UNIMOD representation
    df = replace_modifications(df, modifications["DCCS"], "modified_sequence")
    # Remove underscores and add cysteine carbamidomethylation
    df["modified_sequence"] = df["modified_sequence"].str.replace("_", "")
    df["modified_sequence"] = df["modified_sequence"].str.replace(
        "C", "C[UNIMOD:4]"
    )
    df["task"] = "CCS"

    pickle.dump(df, open(cache_file, "wb"))

    return df


def create_tvt_split(df, test_frac=0.11, val_frac=0.07):
    test_df = df.sample(frac=test_frac)
    test_df = df[df["unmod_seq"].isin(test_df["unmod_seq"])]
    print(f"Test dataset is {len(test_df) * 100 / len(df):.2f}% of total data")

    train_val = df[~df.index.isin(test_df.index)]
    val_df = train_val.sample(frac=val_frac)
    val_df = train_val[train_val["unmod_seq"].isin(val_df["unmod_seq"])]
    print(
        f"Validation dataset is {len(val_df) * 100 / len(df):.2f}% of total data"
    )

    train_df = train_val[~train_val.index.isin(val_df.index)]
    print(
        f"Train dataset is {len(train_df) * 100 / len(df):.2f}% of total data"
    )
    return train_df, val_df, test_df


def create_mtl_5fold_tvt_splits(
    chron_data_path, dccs_data_paths, from_cache=True
):
    save_dir = os.path.join("data", "mtl_5fold_cv")
    os.makedirs(save_dir, exist_ok=True)

    chron_df = process_chronologer(chron_data_path, from_cache)
    dccs_df = process_DCCS(dccs_data_paths, from_cache)

    df = pd.concat([chron_df, dccs_df]).reset_index(drop=True)

    df["unmod_seq"] = get_unmodified_seqs(df)

    for i in range(5):
        train_df, val_df, test_df = create_tvt_split(df)

        train_df.to_csv(
            open(os.path.join(save_dir, f"train_{i}.csv"), "w"),
            columns=[],
            header=False,
        )
        val_df.to_csv(
            open(os.path.join(save_dir, f"val_{i}.csv"), "w"),
            columns=[],
            header=False,
        )
        test_df.to_csv(
            open(os.path.join(save_dir, f"test_{i}.csv"), "w"),
            columns=[],
            header=False,
        )

    del df["unmod_seq"]

    df.to_csv(open(os.path.join(save_dir, "all_data.csv"), "w"))


def create_learning_curve_data(
    chron_data_path, dccs_data_paths, from_cache=True
):
    save_dir = os.path.join("data", "mtl_learning_curves")
    os.makedirs(save_dir, exist_ok=True)

    chron_df = process_chronologer(chron_data_path, from_cache)
    dccs_df = process_DCCS(dccs_data_paths, from_cache)

    df = pd.concat([chron_df, dccs_df]).reset_index(drop=True)
    df["unmod_seq"] = get_unmodified_seqs(df)

    # Create a fixed val and test dataset
    df_train, df_val, df_test = create_tvt_split(df, 0.05, 0.06)

    del df["unmod_seq"]
    df.to_csv(open(os.path.join(save_dir, "all_data.csv"), "w"))

    df_val.to_csv(
        open(os.path.join(save_dir, f"val.csv"), "w"),
        columns=[],
        header=False,
    )
    df_test.to_csv(
        open(os.path.join(save_dir, f"test.csv"), "w"),
        columns=[],
        header=False,
    )

    df_train_chron = df_train[df_train["task"] == "iRT"]
    df_train_dccs = df_train[df_train["task"] == "CCS"]

    data_sizes = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1]

    for chron_size in data_sizes:
        for dccs_size in data_sizes:
            if chron_size == 0 and dccs_size == 0:
                continue
            chron_train = df_train_chron.sample(frac=chron_size)
            dccs_train = df_train_dccs.sample(frac=dccs_size)
            df_train_data_sizes = pd.concat([chron_train, dccs_train])
            df_train_data_sizes.to_csv(
                open(
                    os.path.join(
                        save_dir,
                        f"train_{int(chron_size * 100):02}_{int(dccs_size * 100):02}.csv",
                    ),
                    "w",
                ),
                columns=[],
                header=False,
            )


if __name__ == "__main__":
    create_mtl_5fold_tvt_splits(
        "data/Chronologer_DB_220308.csv",
        ["data/SourceData_Figure_1.csv", "data/SourceData_Figure_4.csv"],
    )
    create_learning_curve_data(
        "data/Chronologer_DB_220308.csv",
        ["data/SourceData_Figure_1.csv", "data/SourceData_Figure_4.csv"],
    )
