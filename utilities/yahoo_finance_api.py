"""yahoo_finance_api.py"""

from tqdm import tqdm  # used to display a progress bar

import pandas as pd
import numpy as np

from yahoo_fin.stock_info import get_data
from functools import reduce

from utilities.general import list_grouper

import logging

# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

MAX_NUM_TICKERS = 30


def get_open_price_df(ticker: {str, list}, start_date=None, end_date=None)->pd.DataFrame:
    """
    Returns a DataFrame with dates as index, ticker_list as columns and open price data as values
    :param ticker: str, list
    :param start_date: str or datetime
    :param end_date: str or datetime
    :return: dict
    """
    return get_ohlc_volume(ticker=ticker, start_date=start_date, end_date=end_date)['open']


def get_high_price_df(ticker: {str, list}, start_date=None, end_date=None)->pd.DataFrame:
    """
    Returns a DataFrame with dates as index, ticker_list as columns and high price data as values
    :param ticker: str, list
    :param start_date: str or datetime
    :param end_date: str or datetime
    :return: dict
    """
    return get_ohlc_volume(ticker=ticker, start_date=start_date, end_date=end_date)['high']


def get_low_price_df(ticker: {str, list}, start_date=None, end_date=None)->pd.DataFrame:
    """
    Returns a DataFrame with dates as index, ticker_list as columns and low price data as values
    :param ticker: str, list
    :param start_date: str or datetime
    :param end_date: str or datetime
    :return: dict
    """
    return get_ohlc_volume(ticker=ticker, start_date=start_date, end_date=end_date)['low']


def get_close_price_df(ticker: {str, list}, start_date=None, end_date=None)->pd.DataFrame:
    """
    Returns a DataFrame with dates as index, ticker_list as columns and close price data as values
    :param ticker: str, list
    :param start_date: str or datetime
    :param end_date: str or datetime
    :return: dict
    """
    return get_ohlc_volume(ticker=ticker, start_date=start_date, end_date=end_date)['close']


def get_adj_close_price_df(ticker: {str, list}, start_date=None, end_date=None)->pd.DataFrame:
    """
    Returns a DataFrame with dates as index, ticker_list as columns and adjusted close price data as values
    :param ticker: str, list
    :param start_date: str or datetime
    :param end_date: str or datetime
    :return: dict
    """
    return get_ohlc_volume(ticker=ticker, start_date=start_date, end_date=end_date)['adjclose']


def get_volume_df(ticker: {str, list}, start_date=None, end_date=None)->pd.DataFrame:
    """
    Returns a DataFrame with dates as index, ticker_list as columns and volume data as values
    :param ticker: str, list
    :param start_date: str or datetime
    :param end_date: str or datetime
    :return: dict
    """
    return get_ohlc_volume(ticker=ticker, start_date=start_date, end_date=end_date)['volume']


def get_ohlc_volume(ticker: {str, list}, start_date=None, end_date=None)->dict:
    """
    Returns a dict with data labels as keys (e.g. 'close') and DataFrame as values where the DataFrame has dates as
    index and ticker_list as columns
    :param ticker: str, list
    :param start_date: str or datetime
    :param end_date: str or datetime
    :return: dict
    """
    # convert the given ticker_list to a list when not already a list
    if not isinstance(ticker, list):
        logger.debug('convert ticker input to a list')
        ticker = [ticker]
    logger.info("download Open, High, Low, Close, Adj. Close and Volume for '%s'" % "', '".join(ticker))

    # loop through each ticker and download the data for each ticker
    logger.debug(f"split the number of ticker_list in a list of sub-lists each with a size of {MAX_NUM_TICKERS}")
    ticker_list_list = list_grouper(iterable=ticker, n=MAX_NUM_TICKERS)
    stock_data_list = []
    tickers_with_no_data = []
    for ticker_list in tqdm(ticker_list_list, total=len(ticker_list_list), desc='Loop through each ticker'):
        try:
            stock_data_list.extend(
                [get_data(ticker=t, start_date=start_date, end_date=end_date) for t in ticker_list]
            )
        except (AssertionError, KeyError):
            # this error is raised when we try to download data when nothing exists (e.g. loading prices of a stock
            # before its IPO).
            # in these cases a column of NaN will be added
            for t in ticker_list:
                try:
                    stock_data_list.append(
                        get_data(ticker=t, start_date=start_date, end_date=end_date)
                    )
                    logger.info(f"'{t}' is OK")
                except (AssertionError, KeyError) as e:
                    error_code = e.args[0]
                    if error_code == 'timestamp':
                        # in cases where the specified time interval does not have any prices
                        pass
                    else:
                        if error_code['chart']['error']['code'] == 'Not Found':
                            raise ValueError(f"'{t}' is not a recognized ticker")
                    logger.info(f"no data available between {start_date} and {end_date} for '{t.upper()}'")
                    tickers_with_no_data.append(t.upper())
                    continue

    logger.debug("combine all the DataFrame into one DataFrame")
    if not len(stock_data_list):
        raise ValueError(f"no data available between {start_date} and {end_date} for '%s'" % "', '".join(ticker))
    comb_df = reduce(lambda x, y: x.append(y), stock_data_list)

    # pivot such that ticker_list are columns and values are the specified data e.g. 'open'
    logger.debug("for each data type, pivot the DataFrame and store the result in a dict")
    result = {}
    data_labels = ['open', 'high', 'low', 'close', 'adjclose', 'volume']
    for d_lbl in data_labels:
        df = comb_df.pivot(columns='ticker', values=d_lbl)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        if len(tickers_with_no_data):
            # when there are ticker_list that does not have any data for the given period, add a column of nan
            df[tickers_with_no_data] = np.nan

        df = df[[t.upper() for t in ticker]].copy()
        result[d_lbl] = df
    return result



