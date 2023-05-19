import pandas as pd


def get_healthy(df):
    """
    This function takes a pandas DataFrame as input and returns
    a subset of the DataFrame containing healthy individuals.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to filter

    Returns:
    pandas.DataFrame: A subset of the original DataFrame containing
                      individuals who meet the filter criteria
    """

    df_healthy = df[
        (df.GOLD_stage == "0")
        & (df.copd_diagnosis == False)
        & (df.asthma_diagnosis == False)
        & (df.cancer_type != "LONGKANKER")
        & (df.cancer_type != "BORST LONG")
    ]
    return df_healthy


def normalise_bps(
    df: pd.DataFrame, bps: list, norm_to: str = "length_at_scan"
) -> pd.DataFrame:
    """
    Normalizes the base pairs (bps) data in a pandas dataframe (df) with respect to a specified column (norm_to).
    Parameters
    ----------
    df : pandas.DataFrame
        A pandas dataframe containing the base pairs data.
    bps : list
        A list of column names in the dataframe to be normalized.
    norm_to : str, optional
        The name of the column to normalize the bps data to. Default is "length_at_scan".

    Returns
    -------
    pandas.DataFrame
        A new pandas dataframe with the normalized bps data.
    """
    # Calculate the normalization factor
    norm_factor = df[norm_to]

    # Normalize the bps data
    for bp in bps:
        df[bp] = df[bp] / norm_factor

    return df


def prettify_names(name: str) -> str:
    """
    This function takes a str of names as input and returns a new str with the names prettified.
    Each name in the input str is modified by removing 'bp_' prefix and capitalizing the first letter of the name.

    :param name: A name str to be prettified.
    :type name: str

    :return: A prettified name.
    :rtype: str

    :raises TypeError: If the input parameter is not a str.
    """
    try:
        if not isinstance(name, str):
            raise TypeError("Input parameter must be a str.")

        return name.replace("bp_", "").replace("_", " ").title()

    except TypeError as error:
        print(error)

# Min-max Normalisation
def min_max_scale(data, param):
    data[param] = (data[param] - data[param].min()) / (data[param].max() - data[param].min())
    return data
