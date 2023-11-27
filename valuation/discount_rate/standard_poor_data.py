"""standard_poor_data.py
These scripts are used to gathering earnings, dividends and buyback data from S&P Global and clean it up as well as
adjust it. This data is used when estimating the implied equity risk premium.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

from utilities.general import get_project_root
from utilities.fred import get_sp500_price


pd.options.mode.chained_assignment = None

BUYBACKS_DATA_URL = r'https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-buyback.xlsx'
INDEX_EARNINGS_DATA_URL = r'https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx'
ACT_REPORTING_CUTOFF = 0.75  # threshold for treating a quarters EPS as actual
HISTORICAL_DIVIDEND_BUYBACK_DATA_PATH = get_project_root() / 'data/historical_dividend_buyback_data.csv'


def get_earnings() -> pd.DataFrame:
    """
    Returns a DataFrame with the quarterly earnings of S&P 500 (actual and bottom-up estimates), indicator for actual or
    estimate and rolling 12 month earnings
    :return: DataFrame
    """
    # load the data from S&P global and store it as a dictionary of two DataFrames
    raw_df_dict = pd.read_excel(INDEX_EARNINGS_DATA_URL,
                                sheet_name=['ESTIMATES&PEs', 'QUARTERLY DATA'])
    est_earnings_df = _get_estimated_earnings(raw_estimates_df=raw_df_dict['ESTIMATES&PEs'])
    act_earnings_df = _get_actual_earnings(raw_actuals_df=raw_df_dict['QUARTERLY DATA'])

    # merge estimates and actual EPS data
    eps_df = pd.concat([est_earnings_df, act_earnings_df], axis=1)
    eps_df = eps_df.groupby(level=0, axis=1).first()
    eps_df['EPS LTM'] = eps_df['EPS'].rolling(window=4).sum()
    return eps_df


def _get_estimated_earnings(raw_estimates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame assumed to be a result of loading data from S&P Global and fetches the earnings estimates
    contained
    :param raw_estimates_df: DataFrame
    :return: DataFrame
    """
    # find the row that contains the estimates
    raw_est_eps_df = raw_estimates_df.iloc[:, [0, 3]]  # sheet 'ESTIMATES&PEs' column A and D
    est_row_idx = list(raw_est_eps_df.index[raw_est_eps_df['S&P Dow Jones Indices'] == 'ESTIMATES'])[0]
    act_row_idx = list(raw_est_eps_df.index[raw_est_eps_df['S&P Dow Jones Indices'] == 'ACTUALS'])[0]
    est_eps_df = raw_est_eps_df.iloc[est_row_idx + 1:act_row_idx - 1, :]
    est_eps_df.columns = ['As of date', 'EPS']
    est_eps_df['Est./Act.'] = 'Est.'

    last_date_str = est_eps_df.iloc[-1, 0]
    if '(' in last_date_str:
        # extract the substring within the ( )
        s = last_date_str[last_date_str.find('(')+1: last_date_str.find(')')]
        # remove the substring after the date
        est_eps_df.iloc[-1, 0] = last_date_str.split(" (")[0]
        if s.lower() == 'prelim.':
            # in some cases it is just '.Prelim'
            # ask user to use it as actual or estimated EPS
            ask_user = True
            while ask_user:
                print(f'Earnings for {last_date_str.split(" (")[0]} is preliminary.')
                prelim_eps_treatment = input('Should it be treated as an estimate? (y/n): ')
                if prelim_eps_treatment.lower() in ['yes', 'y']:
                    est_eps_df.iloc[-1, -1] = 'Est.'
                elif prelim_eps_treatment.lower() in ['no', 'n']:
                    est_eps_df.iloc[-1, -1] = 'Act.'
                else:
                    print(f"'{prelim_eps_treatment}' is not a recognized input\n")
                    continue
                ask_user = False
        else:
            # if the reporting is above the cutoff, treat is as actual, else estimate
            pct_reported = float(s.replace("%", "")) / 100
            if pct_reported >= ACT_REPORTING_CUTOFF:
                est_eps_df.iloc[-1, -1] = 'Act.'
            else:
                est_eps_df.iloc[-1, -1] = 'Est.'

    est_eps_df.set_index('As of date', inplace=True)
    est_eps_df.index = pd.to_datetime(est_eps_df.index)
    return est_eps_df


def _get_actual_earnings(raw_actuals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame assumed to be a result of loading data from S&P Global and fetches the actual earnings contained
    :param raw_actuals_df:
    :return:
    """
    # get the historical actual EPS figures
    act_eps_df = raw_actuals_df.iloc[6:, [0, 2]]
    act_eps_df.dropna(inplace=True)
    act_eps_df.set_index(act_eps_df.columns[0], inplace=True)
    act_eps_df.index = pd.to_datetime(act_eps_df.index)
    act_eps_df.columns = ['EPS']
    act_eps_df['Est./Act.'] = 'Act.'
    return act_eps_df


def get_dividend_buyback_data(use_extended_hist: bool = True) -> pd.DataFrame:
    """
    Reads dividend and buyback data from S&P Global, cleans it up and return result as a DataFrame
    :param use_extended_hist: bool if True, combine the downloaded result with already stored data (the S&P Global data
    only goes back so far)
    :return: DataFrame
    """
    # download data and store as a DataFrame
    df = pd.read_excel(BUYBACKS_DATA_URL, sheet_name='TABLE', skiprows=range(1, 11))
    start_idx = df.iloc[:, 0].isna()[df.iloc[:, 0].isna() == True].index[0] + 1
    buyback_df = df.iloc[start_idx:, :]

    # need to remove a substring from the first date in the calendar
    buyback_df.iloc[0, 0] = buyback_df.iloc[0, 0].split(' ')[0]
    buyback_df.set_index('S&P Dow Jones Indices', inplace=True)
    buyback_df.index = pd.to_datetime(buyback_df.index)
    buyback_df.sort_index(inplace=True)
    buyback_df.columns = ['Market capitalization', 'Operating earnings', 'Reported earnings', 'Dividends', 'Buybacks',
                          'Dividend yield', 'Buyback yield', 'Dividend + Buyback yield']
    buyback_df[['Dividend yield', 'Buyback yield']] = buyback_df[['Dividends', 'Buybacks']] / buyback_df[['Market capitalization']].values
    buyback_df['Dividend + Buyback yield'] = buyback_df[['Dividend yield', 'Buyback yield']].sum(axis=1)

    # trailing 12 months for all values
    ltm_columns = list(buyback_df.columns)
    ltm_columns.remove('Market capitalization')
    buyback_df[ltm_columns] = buyback_df[ltm_columns].rolling(window=4).sum()

    # merge the price column with the buyback data
    spx = get_sp500_price()
    spx.name = 'SPX'
    buyback_df = pd.merge_asof(buyback_df, spx, left_index=True, right_index=True)
    buyback_df['Index point ratio'] = buyback_df['SPX'] / buyback_df['Market capitalization'].values

    # convert nominal values to index points
    nominal_values = ['Operating earnings', 'Reported earnings', 'Dividends', 'Buybacks']
    buyback_df[nominal_values] = buyback_df[nominal_values] * buyback_df[['Index point ratio']].values

    # additional columns: Dividend + Buybacks and Total Payout Ratio
    buyback_df['Dividend + Buybacks'] = buyback_df[['Dividends', 'Buybacks']].sum(axis=1)
    buyback_df['Total payout ratio'] = buyback_df['Dividend + Buybacks'] / buyback_df['Operating earnings'].values

    # make a yearly data frame with the end of year rolling 12 months data
    # for the current year take the latest rolling
    yearly_buyback_df = buyback_df.groupby(buyback_df.index.year).last()

    # drop unnecessary columns
    yearly_buyback_df.drop(['Market capitalization', 'Operating earnings', 'Reported earnings', 'Index point ratio', 'SPX'],
                           inplace=True, axis=1)
    if use_extended_hist:
        yearly_buyback_df = _extend_hist_dividend_buyback(dividend_buyback_df=yearly_buyback_df)
    return yearly_buyback_df


def extended_est_earnings(earnings_df: pd.DataFrame, growth_rates: np.array) -> pd.DataFrame:
    """
    Returns a DataFrame where the last value in the 'EPS LTM' column gets multiplied by 1 + growth rate to estimate the
    forward EPS LTM. The DateTimeIndex gets extended on a yearly basis
    :param earnings_df: DataFrame
    :param growth_rates: np.array
    :return: DataFrame
    """
    ext_ltm_eps = (1 + growth_rates).cumprod() * earnings_df['EPS LTM'].iloc[-1]

    # add new rows containing extended estimates
    # latest_date = eps_df.index[-1]
    ext_ltm_eps_df = pd.DataFrame(
        index=[
            datetime(year=earnings_df.index[-1].year + 1 + i,
                     month=earnings_df.index[-1].month,
                     day=30)
            for i in range(len(ext_ltm_eps))
        ],
        data={
            'EPS': ext_ltm_eps,
            'Est./Act.': 'Est.',
            'EPS LTM': ext_ltm_eps
        }
    )

    earnings_df = earnings_df.append(ext_ltm_eps_df)
    return earnings_df


def overwrite_est_earnings(earnings_df: pd.DataFrame, months_growth_rate_map: dict) -> pd.DataFrame:
    """
    Removes the rows in earnings_df where the 'Est/Act.' column is 'Est.' and replaces the estimates based on a given
    set of growth rates. Assume that one wants the next two estimates to grow with 5% for the next 6 months and then 2%
    for the subsequent 12 months. Then the months_growth_rate_map should be specified as
    {6: 0.05, 18: 0.02}
    :param earnings_df: DataFrame
    :param months_growth_rate_map: dict {key int: value float} with months to next estimate as key and periodic growth
    rate as value
    :return: DataFrame
    """
    # drop all of the rows with an estimate
    earnings_df = earnings_df.loc[~(earnings_df['Est./Act.'] == 'Est.')]

    # store the last date and LTM EPS
    last_date = earnings_df.index[-1]
    eps_ltm = earnings_df['EPS LTM'].iloc[-1]

    # calculate the new dates using the month key and for the est. LTM EPS use the growth value to grow the LTM EPS
    new_dates = [last_date + relativedelta(months=m) for m in months_growth_rate_map.keys()]
    est_eps_ltm = eps_ltm * (1 + np.array(list(months_growth_rate_map.values()))).cumprod()

    # add new rows containing extended estimates
    est_eps_ltm_df = pd.DataFrame(
        index=new_dates,
        data={
            'EPS': est_eps_ltm,
            'Est./Act.': 'Est.',
            'EPS LTM': est_eps_ltm
        }
    )
    earnings_df = earnings_df.append(est_eps_ltm_df)
    return earnings_df


def _extend_hist_dividend_buyback(dividend_buyback_df: pd.DataFrame)->pd.DataFrame:
    """
    Returns a DataFrame that has been merged with already existing data
    :param dividend_buyback_df: DataFrame
    :return: DataFrame
    """
    hist_dividend_buyback_df = pd.read_csv(HISTORICAL_DIVIDEND_BUYBACK_DATA_PATH, index_col=0)

    # reorder columns to have the same order as the dividend_buyback_df
    hist_dividend_buyback_df = hist_dividend_buyback_df[dividend_buyback_df.columns]
    hist_dividend_buyback_df.dropna(inplace=True)

    # remove the rows from the dividend_buyback_df that already are stored as historical values (sometimes the
    # downloaded values only contain data for a part of the year)
    new_dividend_buyback_df = dividend_buyback_df.loc[~dividend_buyback_df.index.isin(hist_dividend_buyback_df.index)]

    # lookup available year - 1
    # hist_dividend_buyback_df = hist_dividend_buyback_df.loc[:dividend_buyback_df.index[0] - 1]
    # combine the two DataFrames
    result = pd.concat([hist_dividend_buyback_df, new_dividend_buyback_df], axis=0)
    result.set_index(result.columns[0], inplace=True)
    return result


