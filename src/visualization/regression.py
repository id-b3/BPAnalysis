#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from data.util.dataframe import min_max_scale


def make_plots(data, bps, out_path):

    sns.set_theme(style="whitegrid")

    data = min_max_scale(data, ["age_at_scan", "length_at_scan", "weight_at_scan", "bmi"] + bps)

    for param in bps:

        for var in ["age_at_scan", "length_at_scan", "weight_at_scan", "bmi"]:
            data_reg = data[[var, param, "smoking_status"]].dropna()
            r, p = stats.pearsonr(data_reg[var], data_reg[param])

            fig = sns.lmplot(data=data_reg,
                             x=var,
                             y=param,
                             hue="smoking_status",
                             truncate=False,
                             scatter=True, scatter_kws={"alpha": 0.3},)
            print("Pearson for {} and {}: {}".format(var, param, r))
            sns.despine(left=True)
            fig.set(ylim=(0,1))
            fig.fig.savefig(f"{str(out_path / param)}_{var}_regression.png", dpi=300)
            plt.close()
