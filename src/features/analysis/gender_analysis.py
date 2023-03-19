#!/usr/bin/env python3

import argparse
import pandas as pd
import os
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Define command line arguments
parser = argparse.ArgumentParser(
    description="Perform one-way ANOVA and Tukey's HS post-hoc test.")
parser.add_argument("file", type=str, help="Path to input DataFrame CSV file.")
parser.add_argument("parameters",
                    type=str,
                    help="Comma separated list of parameters to test.")
parser.add_argument("output",
                    type=str,
                    help="Path to output folder for results CSV file.")
parser.add_argument("--healthy",
                    action="store_true",
                    help="Whether to analyse only the healthy population.")

args = parser.parse_args()

# Load DataFrame
df = pd.read_csv(args.file)

if args.healthy:
    df = df[(df.GOLD_stage == "0")]
    df = df[(df.copd_diagnosis == False)
            & (df.asthma_diagnosis == False)
            & (df.cancer_type != "LONGKANKER") &
            (df.cancer_type != "BORST LONG")]
# Parse parameters to test
parameters = args.parameters.split(",")

# Prepare output folder
os.makedirs(args.output, exist_ok=True)


def get_smoking_status(row):
    if row["current_smoker"] is True:
        return "current_smoker"
    if row["ex_smoker"] is True:
        return "ex_smoker"
    if row["never_smoker"] is True:
        return "never_smoker"
    else:
        return None


df["smoking_status"] = df.apply(get_smoking_status, axis=1)
df = df.dropna(subset=["smoking_status"])


# Function to perform one-way ANOVA and Tukey's test
def perform_anova_and_tukey(dataframe, parameter):
    # One-way ANOVA
    anova_result = stats.f_oneway(
        dataframe[dataframe["smoking_status"] == "current_smoker"][parameter],
        dataframe[dataframe["smoking_status"] == "never_smoker"][parameter],
        dataframe[dataframe["smoking_status"] == "ex_smoker"][parameter])

    # Check if ANOVA is significant
    if anova_result.pvalue < 0.05:
        # Perform Tukey's test
        tukey = pairwise_tukeyhsd(endog=dataframe[parameter],
                                  groups=dataframe["smoking_status"],
                                  alpha=0.05)
        return (True, anova_result, tukey)
    else:
        return (False, anova_result, None)


# Initialize results table
results = []

# Perform tests for each gender and parameter
for gender in ["MALE", "FEMALE"]:
    gender_df = df[df["gender"] == gender]
    for param in parameters:
        significant, anova, tukey = perform_anova_and_tukey(gender_df, param)
        result = {
            "gender": gender,
            "parameter": param,
            "anova_f": anova.statistic,
            "anova_p": anova.pvalue,
            "significant": significant
        }
        if significant:
            group1, group2 = tukey._multicomp.pairindices
            for g1, g2, pvalue, meandiff in zip(group1, group2, tukey.pvalues,
                                                tukey.meandiffs):
                pair = (tukey.groupsunique[g1], tukey.groupsunique[g2])
                result[f"pvalue_{pair[0]}_vs_{pair[1]}"] = pvalue
                result[f"meandiff_{pair[0]}_vs_{pair[1]}"] = meandiff

        results.append(result)

# Convert results to DataFrame and save as CSV
results_df = pd.DataFrame(results)
results_df = results_df.round(4)
output_file = os.path.join(args.output, "gender_anova_tukey_results.csv")
results_df.to_csv(output_file, index=False)