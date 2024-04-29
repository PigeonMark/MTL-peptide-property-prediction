import pandas as pd


def apply_index_file(data_df, i_file):
    df_i = pd.read_csv(open(i_file, "r"), index_col=False, header=None)
    return data_df.iloc[df_i[0]]


def filter_on_tasks(df, tasks):
    return df[df["task"].isin(tasks)]


def read_train_val_test_data(args):
    """
    Read train val and/or test data. Reads all data dataframes for which the arguments were given, otherwise None
    :param args:
    :return:
    """
    if args.use_1_data_file:
        all_data = pd.read_csv(args.data_file, index_col=0)

        df_train = (
            filter_on_tasks(
                (apply_index_file(all_data, args.train_i)),
                args.tasks,
            )
            if args.train_i is not None
            else None
        )
        df_val = (
            filter_on_tasks(
                (apply_index_file(all_data, args.val_i)),
                args.tasks,
            )
            if args.val_i is not None
            else None
        )
        df_test = (
            filter_on_tasks(
                apply_index_file(all_data, args.test_i), args.tasks
            )
            if args.test_i is not None
            else None
        )

    else:
        df_train = filter_on_tasks(
            (
                pd.read_csv(args.train_file, index_col=0)
                if args.train_file is not None
                else None
            ),
            args.tasks,
        )
        df_val = filter_on_tasks(
            (
                pd.read_csv(args.val_file, index_col=0)
                if args.val_file is not None
                else None
            ),
            args.tasks,
        )
        df_test = filter_on_tasks(
            (
                pd.read_csv(args.val_file, index_col=0)
                if args.test_file is not None
                else None
            ),
            args.tasks,
        )

    return df_train, df_val, df_test
