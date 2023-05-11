#!/usr/bin/env python3

import statsmodels.api as sm


def fit_analyse(data, bps, out_path, normalised, logger):
    data["pack_year_categories"] = data["pack_year_categories"].replace(
        "0", "0 pack-years"
    )
    # Perform multivariate linear regression for each dependent variable
    for param in bps:
        # Extract independent variables
        independent_vars = [
            "gender",
            "age_at_scan",
            "length_at_scan",
            "weight_at_scan",
            "current_smoker",
            "pack_year_categories",
        ]

        # Normalising the data
        if normalised:
            for var in independent_vars:
                if data[var].dtype == int or data[var].dtype == float:
                    data[var] = (data[var] - data[var].min()) / (
                        data[var].max() - data[var].min()
                    )
            data[param] = (data[param] - data[param].min()) / (
                data[param].max() - data[param].min()
            )
            output_file = out_path / f"multivariate_report_{param}_normalised.txt"
        else:
            output_file = out_path / f"multivariate_report_{param}.txt"

        # Create formula for the model
        formula = param + " ~ " + " + ".join(independent_vars)

        # Fit the model and print the results
        model = sm.formula.ols(formula=formula, data=data).fit()
        logger.debug(model.summary())

        # Save the results to a text file
        with open(output_file, "w") as f:
            f.write(str(model.summary()))
