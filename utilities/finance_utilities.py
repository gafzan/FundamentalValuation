"""finance_utilities.py"""

import pandas as pd
import numpy as np
import scipy.optimize
from datetime import datetime


def net_present_value(discount_rate: float, values: list, dates: list) -> float:
    """
    Returns the net present value for a schedule of cash flows that is not necessarily periodic.
    Equivalent to Excel's XNPV() function
    Credit: https://stackoverflow.com/questions/8919718/financial-python-library-that-has-xirr-and-xnpv-function
    :param discount_rate: float (discount rate to apply to the cash flows)
    :param values: list of floats (cash flows that corresponds to a schedule of payments in dates)
    :param dates: list of dates (schedule of payment dates that corresponds to the cash flow payments)
    :return: float
    """
    if discount_rate <= -1.0:
        return float('inf')
    d0 = dates[0]  # or min(dates)

    return sum([vi / (1.0 + discount_rate) ** ((di - d0).days / 365.0) for vi, di in zip(values, dates)])


def internal_rate_of_return(values: list, dates: list) -> float:
    """
    Returns the internal rate of return for a schedule of cash flows that is not necessarily periodic.
    Equivalent to Excel's XIRR() function
    Credit: https://stackoverflow.com/questions/8919718/financial-python-library-that-has-xirr-and-xnpv-function
    :param values: list of floats (cash flows that corresponds to a schedule of payments in dates)
    :param dates: list of dates (schedule of payment dates that corresponds to the cash flow payments)
    :return: float
    """
    try:
        return scipy.optimize.newton(lambda r: net_present_value(r, values, dates), 0.0)
    except RuntimeError:  # Failed to converge?
        return scipy.optimize.brentq(lambda r: net_present_value(r, values, dates), -1.0, 1e10)


def implied_discount_rate(values: {list, np.array}, dates: {list, np.array}, target_price: float,
                          terminal_growth_rate: float = None, initial_date: datetime = None) -> float:
    """
    Returns the discount rate that makes the present value of the given cash flows + terminal value equal to the
    specified target price.
    :param values: list or array of floats
    :param dates: list or array of datetimes
    :param target_price: float
    :param terminal_growth_rate: float (optional) calculates a terminal value equal to
        last cash flow * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    :param initial_date: datetime (optional) if not specified, then equal to datetime.now()
    :return: float
    """

    if not initial_date:
        initial_date = datetime.now()

    def cash_flow_pv(discount_rate):
        values_ = values.copy()  # make a copy of the list since we are emending the last element
        if terminal_growth_rate:
            terminal_value = values_[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
            values_[-1] = values_[-1] + terminal_value
        pv = sum([vi / (1.0 + discount_rate) ** ((di - initial_date).days / 365.0) for vi, di in zip(values_, dates)])
        return abs(target_price - pv)
    res = scipy.optimize.minimize_scalar(fun=cash_flow_pv)
    return res.x


def beta(price_df: pd.DataFrame, benchmark_price: {pd.DataFrame, pd.Series}, lag: int,
         log_normal_returns: bool = False)->pd.DataFrame:
    """
    Calculates the rolling beta as the ratio of the covariance and variance. Covariance is calculated on the returns of
    the given price DataFrame and the benchmark. Variance is calculate on the returns of the benchmark. Note that there
    is no lag here for the return calculation. In case one needs to look at the beta for weekly returns one needs to
    first convert the frequency of the price DataFrame of to weekly and then use it as an input into this function.
    :param price_df: DataFrame
    :param benchmark_price: DataFrame or Series
    :param lag: int observation window for the rolling beta calculation
    :param log_normal_returns: bool
    :return: DataFrame
    """

    if isinstance(price_df, pd.Series):
        price_df = price_df.to_frame()

    if 'benchmark' in price_df.columns:
        raise ValueError("can't have a column named 'benchmark' in price_df")

    # convert the benchmark price to a DataFrame if necessary and rename the column
    benchmark_price = benchmark_price.copy()
    if isinstance(benchmark_price, pd.Series):
        benchmark_price = benchmark_price.to_frame()
    benchmark_price.columns = ['benchmark']

    # use log normal returns or arithmetic returns on the merged DataFrame (exact match using the index of price_df)
    return_df = price_return(price_df=price_df.join(benchmark_price), log_normal_returns=log_normal_returns)

    # calculate realized beta as the ratio between the covariance and the valriance
    covariance_df = return_df.rolling(window=lag).cov(return_df['benchmark'])
    variance_df = return_df['benchmark'].rolling(window=lag).var()
    beta_df = covariance_df.divide(variance_df, axis='index')
    beta_df.drop('benchmark', axis=1, inplace=True)
    return beta_df


def price_return(price_df: pd.DataFrame, log_normal_returns: bool, lag: int = 1)->pd.DataFrame:
    """
    Returns a DataFrame with arithmetic or logarithmic return
    :param price_df: DataFrame
    :param log_normal_returns: bool
    :param lag: int
    :return: DataFrame
    """
    # either use log normal returns or arithmetic returns
    if log_normal_returns:
        return_df = np.log(price_df) - np.log(price_df.shift(lag))
    else:
        # by default the fill method is forward fill of the price before the return calculation so NaN are set to 0
        return_df = price_df.pct_change(periods=lag, fill_method=None)
    return return_df


def market_value_bond(coupon_rate: float, discount_rate: float, periods_to_maturity: int) -> float:
    """
    Returns the market value of a bond as a percentage of its face value
    :param coupon_rate: float
    :param discount_rate: float
    :param periods_to_maturity: int
    :return: float
    """
    return (1 + discount_rate) ** (-periods_to_maturity) + coupon_rate * (1 - (1 + discount_rate) ** (-periods_to_maturity)) / discount_rate
