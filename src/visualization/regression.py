#!/usr/bin/env python3

import csv
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy.stats as stats

from data.util.dataframe import min_max_scale


def make_plots(data, bps, out_path):

    sns.set_theme(style="whitegrid")
    fit_tests = []

    data = min_max_scale(data, "age_at_scan")

    for param in bps:
        data = min_max_scale(data, param)
        for status in ["never_smoker", "ex_smoker", "current_smoker"]:
            # fit linear regression model
            model_linear = smf.ols(f'{param} ~ age_at_scan', data=data[data[status] == True]).fit()
            sse_linear = model_linear.ssr

            # fit polynomial regression model
            model_poly = smf.ols(f'{param} ~ age_at_scan + I(age_at_scan**2)', data=data[data[status] == True]).fit()
            sse_poly = model_poly.ssr

            # perform F-test
            n = len(data)
            k_linear = 2
            k_poly = 3
            alpha = 0.00238
            f_stat = ((sse_linear - sse_poly) / (k_poly - k_linear)) / \
                (sse_poly / (n - k_poly))
            f_crit = stats.f.ppf(1 - alpha, k_poly - k_linear, n - k_poly)

            f_result = [param, status, sse_linear, sse_poly, f_stat, f_crit]

            if f_stat > f_crit:
                print(f'Polynomial model is a better fit for {param} for {status}')
                order = 1
                f_result.append("poly")
            else:
                print(f'Linear model is sufficient for {param} for {status}')
                order = 1
                f_result.append("linear")
            fit_tests.append(f_result)

        fig = sns.lmplot(data=data,
                         x="age_at_scan",
                         y=param,
                         hue="smoking_status",
                         order=order,
                         truncate=False,
                         scatter=False)
        sns.despine(left=True)
        fig.fig.savefig(f"{str(out_path / param)}_regression.png", dpi=300)
        plt.close()

        with open(f"{str(out_path)}/regression_fit_comparisons.csv",
                  mode="w",
                  newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([
                "parameter", "smoking_group", "sse_linear", "sse_poly", "f_stat", "f_crit", "better model"
            ])
            writer.writerows(fit_tests)
