"""IMPLIED_ERP.py
Run the main() to estimate the implied equity risk premium
"""

import pandas as pd
import numpy as np
from datetime import datetime

from valuation.discount_rate.standard_poor_data import get_earnings
from valuation.discount_rate.standard_poor_data import extended_est_earnings
from valuation.discount_rate.standard_poor_data import overwrite_est_earnings
from valuation.discount_rate.standard_poor_data import get_dividend_buyback_data
from utilities.fred import get_us_10y_cms
from utilities.fred import get_sp500_price
from utilities.finance_utilities import implied_discount_rate


def get_initial_cash_flow(dividend_buyback_df: pd.DataFrame, avg_lag_yrs: int, method: str)->float:
    """
    Returns a float as the estimated cash flow of the S&P 500 either based on historical dividends + buybacks or the
    total gross yield (dividend + buyback yield) multiplied by the latest
    :param dividend_buyback_df: DataFrame
    :param avg_lag_yrs: int
    :param method: str 'yield' or 'amount'
    :return: float
    """
    method = method.lower()
    if method == 'yield':
        spx_latest_price = get_sp500_price().iloc[-1]
        # calculate the initial cash flow based on the historical yield and current level of S&P 500
        initial_cash_flow = dividend_buyback_df['Dividend + Buyback yield'].rolling(window=avg_lag_yrs).mean().iloc[
                                -1] * spx_latest_price
    elif method == 'amount':
        initial_cash_flow = dividend_buyback_df['Dividend + Buybacks'].rolling(window=avg_lag_yrs).mean().iloc[-1]
    else:
        raise ValueError(f"'{method}' is not a recognized method. Can either be 'yield' or 'amount'")

    return initial_cash_flow


def get_terminal_payout_ratio(dividend_buyback_df: pd.DataFrame, avg_lag_yrs: int = None, pct_ile: float = None) -> float:
    """
    Returns a float as the estimated terminal total payout ratio of the S&P 500 either based historical average or a
    percentile
    :param dividend_buyback_df: DataFrame
    :param avg_lag_yrs: int
    :param pct_ile: float
    :return: float
    """
    if [avg_lag_yrs, pct_ile].count(None) == 0 or [avg_lag_yrs, pct_ile].count(None) == 2:
        raise ValueError('One and only one of avg_lag_yrs and pct_ile should be specified')
    if avg_lag_yrs:
        terminal_payout_ratio = dividend_buyback_df['Total payout ratio'].rolling(window=avg_lag_yrs).mean().iloc[-1]
    else:
        terminal_payout_ratio = dividend_buyback_df['Total payout ratio'].quantile(pct_ile)
    return terminal_payout_ratio


def calculate_cash_flow(estimated_earnings: pd.DataFrame, initial_cash_flow: float,
                        terminal_payout_ratio: float) -> pd.DataFrame:
    """
    Returns a DataFrame with the various cash flow components as columns
    :param estimated_earnings: DataFrame
    :param initial_cash_flow: float
    :param terminal_payout_ratio: float
    :return: DataFrame
    """
    cash_flow_df = estimated_earnings.copy()
    interp_payout_ratio = np.linspace(start=initial_cash_flow / cash_flow_df['EPS'].iloc[-1],
                                      stop=terminal_payout_ratio,
                                      num=cash_flow_df.shape[0])
    cash_flow_df['Total payout ratio'] = interp_payout_ratio
    cash_flow_df['Distributions'] = cash_flow_df['Total payout ratio'] * cash_flow_df['EPS'].values
    res = cash_flow_df[['Distributions']]
    res.sort_index(inplace=True)
    return res


def main():
    # ------------------------------------------------------------------------------------------------------------------
    # define the parameters
    est_horizon_yrs = 5
    avg_lag_yrs = 5  # used to estimate initial cash flow and terminal payout ratio
    # pct_ile = 0.75  # used to estimate terminal payout ratio
    top_down_eps_est = False
    month_delta_growth_top_down = {6: 0.0381, 18: 0.11111, 30: 0.08}  # as of 2 Oct 2023 (used when top_down_eps_est is True)
    # growth rates are gathered from Dr. Yardeni's website (https://www.yardeni.com/) 'YRI Forecast: S&P 500 Earnings'
    # Assume that one wants the next two estimates of EPS LTM to grow with 5% for the next 6 months (in the case when
    # latest actual EPS is for Q2) and then 2% for the following 12 months.
    # Then month_delta_growth should be specified as {6: 0.05, 18: 0.02}

    # ------------------------------------------------------------------------------------------------------------------
    # load market data from FRED
    spx_latest_price = get_sp500_price().iloc[-1]
    risk_free_rate = get_us_10y_cms().iloc[-1]
    terminal_growth_rate = risk_free_rate

    # load S&P 500 earnings, dividend and buyback data
    eps_df = get_earnings()
    dividend_buyback_df = get_dividend_buyback_data()

    # here you can overwrite the earnings estimates to not be bottom-up but instead be top-down (it is a bit manual)
    if top_down_eps_est:
        eps_df = overwrite_est_earnings(earnings_df=eps_df, months_growth_rate_map=month_delta_growth_top_down)

    # calculate extended estimate of EPS LTM assuming that the final estimate of yearly growth available converges to
    # the terminal growth rate
    additional_years = est_horizon_yrs - (
            eps_df[eps_df['Est./Act.'] == 'Est.'].index[-1].year - eps_df[eps_df['Est./Act.'] == 'Est.'].index[0].year)
    interp_growth_rates = np.linspace(start=eps_df['EPS LTM'].pct_change(4)[-1], stop=terminal_growth_rate,
                                      num=additional_years + 1)[1:]  # ignore the first growth rate
    eps_df = extended_est_earnings(earnings_df=eps_df, growth_rates=interp_growth_rates)

    # create a DataFrame with all the estimated earnings summed over the fiscal year
    est_eps_df = eps_df[eps_df['Est./Act.'] == 'Est.'].groupby(eps_df[eps_df['Est./Act.'] == 'Est.'].index.year).sum()[
        ['EPS']]
    est_eps_df.index = pd.to_datetime([datetime(year=year, month=12, day=31) for year in est_eps_df.index])

    # calculate the cash flows by first estimating the initial cash flow (using the specified assumptions), estimating
    # a total payout ratio (using the specified assumptions)
    initial_cash_flow = get_initial_cash_flow(dividend_buyback_df=dividend_buyback_df, avg_lag_yrs=avg_lag_yrs,
                                              method='amount')
    terminal_payout_ratio = get_terminal_payout_ratio(dividend_buyback_df=dividend_buyback_df, avg_lag_yrs=avg_lag_yrs)
    cash_flows = calculate_cash_flow(estimated_earnings=est_eps_df, initial_cash_flow=initial_cash_flow,
                                     terminal_payout_ratio=terminal_payout_ratio)

    # solve for the discount rate that brings the present value of the future cash flows to be equal to the current
    # price of the S&P 500
    irr = implied_discount_rate(values=cash_flows['Distributions'].values, dates=cash_flows.index,
                                target_price=spx_latest_price, terminal_growth_rate=terminal_growth_rate)
    print(f'Implied discount rate: {round(irr * 100, 2)}%')
    print(f'Risk free rate: {round(risk_free_rate * 100, 2)}%')
    print(f'Implied equity risk premium: {round((irr - risk_free_rate) * 100, 2)}%')
    print(f"Assuming {'top-down' if top_down_eps_est else 'bottom-up'} EPS estimates")


if __name__ == '__main__':
    main()
