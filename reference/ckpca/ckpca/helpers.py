import pandas as pd
import numpy as np
from typing import Optional

def load_characteristics_data_from_cache(
    parameters: dict,
    data_path: str = 'data'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads characteristics and asset returns data from cache and applies time filters.

    This function loads the characteristics and asset returns data from cached pickle files
    based on the provided parameters. It then applies time filters to both datasets according
    to the specified sample start and end dates.

    Args:
        parameters (dict): A dictionary containing the following keys:
            - 'characteristics_model' (str): The model name for characteristics data.
            - 'sample_start_date' (str): The start date for the sample period.
            - 'sample_end_date' (str): The end date for the sample period.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - asset_rets (pd.DataFrame): The filtered asset returns data.
            - characteristics (pd.DataFrame): The filtered characteristics data.
    """
    asset_rets = pd.read_pickle(
        f'{data_path}/asset_rets_dummy.p'
    )
    characteristics = pd.read_pickle(
        f'{data_path}/characteristics_{parameters["characteristics_model"]}.p'
    )

    # Time filter on characteristics
    mask = (
        (characteristics.index.get_level_values(1) >= parameters['sample_start_date']) &
        (characteristics.index.get_level_values(1) <= parameters['sample_end_date'])
    )
    characteristics = characteristics.loc[mask].copy()

    # Time filter on asset returns
    mask = (
        (asset_rets.index.get_level_values(1) >= parameters['sample_start_date']) &
        (asset_rets.index.get_level_values(1) <= parameters['sample_end_date'])
    )
    asset_rets = asset_rets.loc[mask].copy()

    return asset_rets, characteristics

def train_test_split_double_index(
    data: pd.DataFrame, 
    split_date: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training and testing sets based on a split date.

    This function splits a DataFrame with a multi-level index into training and testing sets.
    The split is performed based on the second level of the index, which is assumed to be a date.
    Optionally, the data can be filtered to include only rows within a specified date range.

    Args:
        data (pd.DataFrame): The input data with a multi-level index.
        split_date (str): The date used to split the data into training and testing sets.
        start_date (Optional[str], optional): The start date for filtering the data. Defaults to None.
        end_date (Optional[str], optional): The end date for filtering the data. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing sets.
    """
    if start_date is not None:
        data = data[
            data.index.get_level_values(1) >= start_date
        ]
    
    if end_date is not None:
        data = data[
            data.index.get_level_values(1) <= end_date
        ]
    
    train = data[
        data.index.get_level_values(1) <= split_date
    ]
    test = data[
        data.index.get_level_values(1) > split_date
    ]

    return train, test


def rotate_rets(rets, rets_is=None):
    anomalies = ['PC' + str(i) for i in range(1, len(rets.columns) + 1)]
    
    if rets_is is None:  # If no in-sample data is provided, use rets
        rets_is = rets

    _, _, Q = np.linalg.svd(np.cov(rets_is, rowvar=False))
    Q = pd.DataFrame(Q, columns=rets.columns, index=anomalies)

    re_rotated = rets @ Q.T
    return pd.DataFrame(re_rotated, index=rets.index, columns=anomalies)
