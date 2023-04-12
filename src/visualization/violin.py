#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.data.subgroup import get_healthy


def main(args):
    df = pd.read_csv(args.in_db)
    if args.healthy:
        df = get_healthy(df)

    df.columns = df.columns.str.replace('_', ' ').str.title()

    bps = args.param_list.replace('_', ' ').title().split(',')

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    for param in bps:
        fig = sns.violinplot(data=df,
                             x="Smoking Status",
                             y=param,
                             hue="Gender",
                             split=True,
                             inner="quart",
                             linewidth=1.5,
                             palette={
                                 "Male": "b",
                                 "Female": "salmon"
                             })
        sns.despine(left=True)
        fig.get_figure().savefig(f"{str(out_path / param)}.png", dpi=300)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_db", type=str, help="Input dfbase csv")
    parser.add_argument("out_dir",
                        type=str,
                        help="Output folder for violin plots.")
    parser.add_argument("param_list",
                        type=str,
                        help="Comma separated list of params to process.")
    parser.add_argument("--healthy",
                        action="store_true",
                        help="Only analyse healthy")
    args = parser.parse_args()
    main(args)
