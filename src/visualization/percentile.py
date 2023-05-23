#!/usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .prettifiers import prettify_axes


def make_plots(data, bps, out_path):
    out_path = out_path / "percentile"
    out_path.mkdir(parents=True, exist_ok=True)

    # Create age categories
    age_label_2 = [
        40,
        42,
        44,
        46,
        48,
        50,
        52,
        54,
        56,
        58,
        60,
        62,
        64,
        66,
        68,
        70,
        72,
        74,
        76,
        78,
        80,
        82,
    ]
    age_cut_2 = np.linspace(40, 88, 22)
    age_cut_2 = np.append(age_cut_2, 100)

    data["age_2yr"] = pd.cut(data["age_at_scan"],
                             bins=age_cut_2,
                             labels=age_label_2,
                             right=False)

    sns.set_theme(style="whitegrid")

    for param in bps:
        for gender in ["Male", "Female"]:
            for sm_stat in ["never_smoker", "ex_smoker", "current_smoker"]:
                percentiles = (data[(data["smoking_status"] == sm_stat)
                                    & (data["gender"] == gender)].groupby(
                                        "age_2yr")[param].quantile(
                                            [0.1, 0.3, 0.5, 0.7,
                                             0.9]).reset_index())
                percentiles = percentiles.rename(
                    columns={"level_1": "Percentile"})
                percentiles.Percentile = percentiles["Percentile"].apply(
                    lambda x: f"{x * 100:.0f}%")

                fig = sns.lmplot(
                    data=percentiles,
                    x="age_2yr",
                    y=param,
                    hue="Percentile",
                    scatter=False,
                    truncate=False,
                    palette=sns.color_palette([
                        "deepskyblue", "mediumseagreen", "green", "orange",
                        "red"
                    ]),
                    ci=None,
                    # order=2,
                    robust=True,
                    line_kws={"alpha": 0.5},
                )

                prettify_axes(fig)

                sns.move_legend(fig,
                                "lower center",
                                bbox_to_anchor=(0.5, 1.0),
                                ncol=5,
                                title=None,
                                frameon=False)

                # Additional customization of the x-axis
                plt.xlabel("Age")
                plt.ylabel(param.replace("bp_", ""))
                plt.title(f"{gender.title()} {sm_stat}")
                plt.tight_layout()

                fig.savefig(f"{str(out_path / param)}_{gender}_{sm_stat}.png",
                            dpi=300)
                plt.close()
