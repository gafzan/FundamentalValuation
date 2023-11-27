"""fred.py"""

import pandas as pd
import os
from fredapi import Fred
import logging

# logger
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


FRED_API_KEY = os.environ.get('FRED_API_KEY')
fred = Fred(api_key=FRED_API_KEY)


def get_us_10y_cms() -> pd.DataFrame:
    """
    Returns US 10Y CMS as a DataFrame with nan filled forward
    :return: DataFrame
    """
    logger.info('Downloading US 10Y CMS from FRED')
    us_10y_cms_data = fred.get_series('DGS10')
    us_10y_cms_data.fillna(method='ffill', inplace=True)
    us_10y_cms_data /= 100.0
    return us_10y_cms_data


def get_sp500_price():
    """
    Returns S&P 500 as a DataFrame with nan filled forward
    :return: DataFrame
    """
    logger.info('Downloading S&P 500 from FRED')
    spx = fred.get_series('SP500')
    spx.fillna(method='ffill', inplace=True)
    return spx


def get_us_corp_option_adjusted_spread(interpolate_additional_ratings: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with US corp. option adjusted spreads
    :param interpolate_additional_ratings: if True, adds additional spreads for ratings with +/-
    :return: DataFrame
    """
    score_ticker_map = {
        'AAA': 'BAMLC0A1CAAA',
        'AA': 'BAMLC0A2CAA',
        'A': 'BAMLC0A3CA',
        'BBB': 'BAMLC0A4CBBB',
        'BB': 'BAMLH0A1HYBB',
        'B': 'BAMLH0A2HYB',
        'CCC': 'BAMLH0A3HYC'
    }
    df = None
    for key, value in score_ticker_map.items():
        logger.info(f'Downloading {key} US Corp. option adjusted spread ({value}) from FRED')
        credit_spread = fred.get_series(value)
        credit_spread = credit_spread.to_frame(key)
        if df is None:
            df = credit_spread
        else:
            df = df.merge(credit_spread, left_index=True, right_index=True, how='outer')

    df /= 100.0

    if interpolate_additional_ratings:
        _interpolate_spread_additional_ratings(credit_spread_df=df)
    return df


def _interpolate_spread_additional_ratings(credit_spread_df: pd.DataFrame) -> None:
    """
    Adds new columns with additional ratings like AA+ and AA- that are simple averages between the available data
    Note that this is not a great approximation since AA- and A+ will have the same spread which is the average of the
    spreads of AA- and A+
    :param credit_spread_df: DataFrame
    :return: None
    """

    new_col_map = {
        'AA+': ['AAA', 'AA'],
        'AA-': ['AA', 'A'],
        'A+': ['AA', 'A'],
        'A-': ['A', 'BBB'],
        'BBB+': ['A', 'BBB'],
        'BBB-': ['BBB', 'BB'],
        'BB+': ['BBB', 'BB'],
        'BB-': ['BB', 'B'],
        'B+': ['BB', 'B'],
        'B-': ['B', 'CCC'],
        'CCC+': ['B', 'CCC']
    }

    for new_col, near_cols in new_col_map.items():
        credit_spread_df[new_col] = credit_spread_df[near_cols].mean(axis=1)
    return

