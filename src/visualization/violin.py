#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt


def make_plots(data, bps, out_path):

    sns.set_theme(style="whitegrid")
    for param in bps:
        fig = sns.violinplot(data=data,
                             x="smoking_status",
                             y=param,
                             hue="gender",
                             split=True,
                             inner="quart",
                             linewidth=1.5,
                             palette={
                                 "Male": "b",
                                 "Female": "salmon"
                             })
        sns.despine(left=True)
        fig.get_figure().savefig(f"{str(out_path / param)}_violin.png", dpi=300)
        plt.close()
