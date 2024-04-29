import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from calculate_metrics import (
    prosit_5cv_irt_metrics,
    dccs_5cv_irt_metrics,
    mtl_transformer_5cv_metrics,
    get_stl_lc_metrics,
)


def plot_cv(results, models, metric="mae", m_name="MAE", title="", show=False):
    data = {metric: [], "model": []}
    for result, model in zip(results, models):
        data[metric] += [r[metric] for r in result]
        data["model"] += [model] * len(result)
    data = pd.DataFrame(data)

    plt.figure(figsize=(5, 3.5))
    sns.swarmplot(data=data, x="model", y=metric, c="black")
    sns.boxplot(data=data, x="model", y=metric, showfliers=False)
    sns.despine()
    plt.xlabel(None)
    plt.ylabel(m_name)
    plt.tight_layout()

    os.makedirs("plots/5foldcv", exist_ok=True)
    plt.savefig(os.path.join("plots", "5foldcv", title), dpi=600)
    if show:
        plt.show()


def plot_stl_lc_curves(show=False, cached=True):
    for log_t, t in zip(["rt", "ccs"], ["iRT", "CCS"]):
        metric_dict = get_stl_lc_metrics(f"stl_{log_t}_lc", cached=cached)
        for metric, m_name in zip(["mae", "d95", "r"], ["MAE", "Δ95%", "R"]):
            data = {f"{t}_frac": [], metric: []}
            for t_frac, result in metric_dict.items():
                t_frac = int(t_frac)
                data[f"{t}_frac"].append(t_frac)
                data[metric].append(result[t][metric])

            data = pd.DataFrame(data)

            plt.figure(figsize=(4, 3))
            sns.lineplot(data=data, x=f"{t}_frac", y=metric)

            plt.ylabel(m_name)
            plt.xlabel(f"{t} data fraction (%)")
            sns.despine()
            plt.tight_layout()

            os.makedirs("plots/lc", exist_ok=True)
            plt.savefig(
                os.path.join("plots", "lc", f"stl_{t}_{metric}"), dpi=600
            )
            if show:
                plt.show()


def plot_5fold_cv_pt_comparison(show=False):
    ref_results = {
        "iRT": prosit_5cv_irt_metrics(),
        "CCS": dccs_5cv_irt_metrics(),
    }
    ref_names = {"iRT": "Prosit", "CCS": "DCCS"}
    mtl_sn_results = mtl_transformer_5cv_metrics(
        "mtl_5foldcv_supervised_none", cached=True
    )
    mtl_ft_results = mtl_transformer_5cv_metrics(
        "mtl_5foldcv_finetune_tape", cached=True
    )
    mtl_fo_results = mtl_transformer_5cv_metrics(
        "mtl_5foldcv_finetune_own", cached=True
    )
    for metric, m_name in zip(["mae", "d95", "r"], ["MAE", "Δ95%", "R"]):
        for t in ["iRT", "CCS"]:
            plot_cv(
                [
                    ref_results[t],
                    mtl_sn_results[t],
                    mtl_fo_results[t],
                    mtl_ft_results[t],
                ],
                [
                    ref_names[t],
                    "MTL Trans-\nformer from\nscratch",
                    "MTL Trans-\nformer fine-\ntune own",
                    "MTL Trans-\nformer fine-\ntune TAPE",
                ],
                metric=metric,
                m_name=m_name,
                title=f"5 fold cv pretrain comparison {t} {metric}",
                show=show,
            )


def plot_5fold_cv_mtl_comparison(show=False):
    ref_results = {
        "iRT": prosit_5cv_irt_metrics(),
        "CCS": dccs_5cv_irt_metrics(),
    }
    ref_names = {"iRT": "Prosit", "CCS": "DCCS"}
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

    for metric, m_name in zip(["mae", "d95", "r"], ["MAE", "Δ95%", "R"]):
        for t in ["iRT", "CCS"]:
            plot_cv(
                [
                    ref_results[t],
                    mtl_ft_results[t],
                    stl_results[t][t],
                ],
                [
                    ref_names[t],
                    "MTL Transformer\nfine-tune TAPE",
                    "STL Transformer\nfine-tune TAPE",
                ],
                metric=metric,
                m_name=m_name,
                title=f"5 fold cv mtl comparison {t} {metric}",
                show=show,
            )


if __name__ == "__main__":
    sns.set_style("whitegrid")
    sns.set_palette(
        sns.color_palette(["#6d9eeb", "#f6b26b", "#93c47d", "#c27ba0"])
    )
    plot_5fold_cv_pt_comparison()
    plot_5fold_cv_mtl_comparison()
    plot_stl_lc_curves()
