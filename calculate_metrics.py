import os.path
from collections import defaultdict

import pandas as pd
from scipy import stats

from src.util import (
    get_log_df,
    config_filter,
    calculate_metrics,
    Result,
)


def read_prosit_result(predict_path):
    prosit_df = pd.read_csv(predict_path)
    return Result(
        prosit_df["predictions"], prosit_df["label"], run=predict_path
    )


def read_dccs_result(predict_path, all_data_path):
    all_data = pd.read_csv(
        all_data_path,
        index_col=0,
        dtype={"Charge": "float64", "DCCS_sequence": str},
    )
    dccs_df = pd.read_csv(predict_path, index_col=0)
    true_ccs = all_data.iloc[dccs_df.index]["label"]
    return Result(dccs_df["predictions"], true_ccs, run=predict_path)


def dccs_5cv_irt_metrics():
    results = []
    for cv in range(5):
        path = f"dccs_results/cv{cv}/out/predictions.csv"
        results.append(
            read_dccs_result(path, "data/mtl_5fold_cv/all_data.csv")
        )
    return results


def prosit_5cv_irt_metrics():
    results = []
    for cv in range(5):
        path = f"prosit_results/mtl_cv{cv}/test_predictions.csv"
        results.append(read_prosit_result(path))
    return results


def mtl_transformer_5cv_metrics(config, cached=True):
    log_df = get_log_df(config_filter(config, exact=False), cached=cached)
    runs = log_df["run"].unique()
    print(f"Found following runs: {runs}")

    task_results = defaultdict(list)
    for run in runs:
        try:
            task_metrics = calculate_metrics(
                os.path.join("lightning_logs", run, "predictions"),
                predict_name="test",
            )
        except FileNotFoundError:
            print(f"No test predictions found for {run}, skipping...")
            continue

        for t, r in task_metrics.items():
            task_results[t].append(r)

    return task_results


def get_stl_lc_metrics(config, cached=True):
    log_df = get_log_df(config_filter(config, exact=False), cached=cached)
    runs = log_df["run"].unique()
    metric_dict = {}
    for t_frac in ["05", "10", "25", "50", "75", "100"]:
        possible_runs = [r for r in runs if f"{config}_{t_frac}," in r]
        if len(possible_runs) > 1:
            print(
                f"Found multiple runs for {config}_{t_frac}: {possible_runs}"
            )
            continue
        if len(possible_runs) == 0:
            print(f"Found no runs for {config}_{t_frac}")
            continue

        run = possible_runs[0]

        try:
            task_metrics = calculate_metrics(
                os.path.join("lightning_logs", run, "predictions"),
                predict_name="test",
            )
        except FileNotFoundError:
            print(f"No test predictions found for {run}, skipping...")
            continue

        metric_dict[t_frac] = task_metrics
    return metric_dict


def calculate_5fold_cv_pt_significance():
    ref_results = {
        "iRT": prosit_5cv_irt_metrics(),
        "CCS": dccs_5cv_irt_metrics(),
    }
    mtl_sn_results = mtl_transformer_5cv_metrics(
        "mtl_5foldcv_supervised_none", cached=True
    )
    mtl_ft_results = mtl_transformer_5cv_metrics(
        "mtl_5foldcv_finetune_tape", cached=True
    )
    mtl_fo_results = mtl_transformer_5cv_metrics(
        "mtl_5foldcv_finetune_own", cached=True
    )

    for t in ["iRT", "CCS"]:
        ress = [
            ref_results[t],
            mtl_sn_results[t],
            mtl_fo_results[t],
            mtl_ft_results[t],
        ]
        names = ["Reference", "from scratch", "finetune own", "finetune tape"]
        for i in range(len(ress)):
            for ii in range(i + 1, len(ress)):
                res_i = [r.mae() for r in ress[i]]
                res_ii = [r.mae() for r in ress[ii]]

                res = stats.mannwhitneyu(res_i, res_ii)
                print(
                    f"{t} p={res.pvalue:.4f}, stat={res.statistic:.4f} between {names[i]} and {names[ii]}"
                )


def calculate_5fold_cv_mtl_significance(two_sided=True):
    ref_results = {
        "iRT": prosit_5cv_irt_metrics(),
        "CCS": dccs_5cv_irt_metrics(),
    }
    mtl_ft_results = mtl_transformer_5cv_metrics(
        "mtl_5foldcv_finetune_tape", cached=True
    )
    stl_results = {
        "iRT": mtl_transformer_5cv_metrics(
            "stl_iRT_5foldcv_finetune_tape", cached=True
        ),
        "CCS": mtl_transformer_5cv_metrics(
            "stl_CCS_5foldcv_finetune_tape", cached=True
        ),
    }
    for t in ["iRT", "CCS"]:
        ress = [
            ref_results[t],
            mtl_ft_results[t],
            stl_results[t][t],
        ]
        names = ["Reference", "MTL finetune tape", "STL finetune tape"]
        for i in range(len(ress)):
            if two_sided:
                from_ = i + 1
            else:
                from_ = 0
            for ii in range(from_, len(ress)):
                res_i = [r.mae() for r in ress[i]]
                res_ii = [r.mae() for r in ress[ii]]

                res = stats.mannwhitneyu(
                    res_i,
                    res_ii,
                    alternative="two-sided" if two_sided else "greater",
                )
                if two_sided:
                    print(
                        f"{t} p={res.pvalue:.5f}, stat={res.statistic:.5f} between {names[i]} and {names[ii]}"
                    )
                else:
                    print(
                        f"{t} p={res.pvalue:.5f}, stat={res.statistic:.5f} that {names[i]} is greater than {names[ii]}"
                    )


if __name__ == "__main__":
    calculate_5fold_cv_pt_significance()
    calculate_5fold_cv_mtl_significance(two_sided=True)
    calculate_5fold_cv_mtl_significance(two_sided=False)
