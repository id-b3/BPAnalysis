#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm


def fit_analyse(data: pd.DataFrame, bps: list, i_var: str, out_path: Path):
    """
    This function performs univariate analysis on a given data frame.

    Parameters:
    data (pd.DataFrame): The data frame to perform the analysis on.
    bps (list): A list of parameters to loop through and calculate Pearson's cc and R-squared.
    i_var (str): The independent variable to calculate correlation against.
    out_path (Path): The output file path where the results will be saved in CSV format.

    Returns:
    None

    Raises:
    None
    """

    results = []

    for gender in ["Male", "Female"]:
        gender_data = data[data["gender"] == gender]

        # Loop parameters and calculate Pearson's cc and R-squared
        for param in bps:
            for group in data["smoking_status"].unique():
                data_param = gender_data.dropna(subset=[param, i_var])
                data_param = data_param[data_param.smoking_status == group]
                print(f"Calculating {param} wrt {i_var} for {gender} {group}")
                X = data_param[[i_var]].values
                y = data_param[[param]].values
                pearson, _ = pearsonr(X.T[0], y.T[0])
                model = sm.OLS(y, sm.add_constant(X)).fit()
                rsquared = model.rsquared
                pval = round(model.pvalues[1], 4)

                # Create a dictionary to store results
                result = {
                    "Group": f"{gender}_{group}",
                    "Parameter": param,
                    "Pearson Correlation": pearson,
                    "R-squared": rsquared,
                    "P-value": pval,
                }
                results.append(result)

    # Create a pandas DataFrame from the results dictionary
    results_df = pd.DataFrame.from_dict(results)
    results_df = results_df.round(2)

    # Output results to a CSV file
    results_df.to_csv((out_path / f"univariate_analysis_wrt_{i_var}.csv"), index=False)
