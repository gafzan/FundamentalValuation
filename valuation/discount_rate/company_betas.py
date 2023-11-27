"""company_betas.py
Run main to estimate betas for all stocks stored in 'BETA_INPUT_US_STOCK_PATH' csv and then to aggregate betas based on
industry. The result of the industry aggregates are stored in 'INDUSTRY_BETA_PATH'

historical_beta() can be called to get the beta estimates for one or more Yahoo tickers
get_industry_beta() returns the industry beta (levered or unlevered) for a given set of industry weights
get_cash_adjusted_industry_beta() returns the cash-adjusted unlevered industry beta for a given set of industry weights

Assumes that the data stored in BETA_INPUT_US_STOCK_PATH contains Ticker, Industry, Total Debt (LTM),
Cash/ST Investments (LTM), Effective Tax Rate - (Ratio) (LTM)
I get this data from koyfin.com
"""

import pandas as pd
from datetime import datetime
from datetime import timedelta

import logging

from utilities.general import list_grouper
from utilities.yahoo_finance_api import get_adj_close_price_df
from utilities.general import get_project_root

ADJUSTED_TICKER_MAP = {
    'BFB': 'BF-B',
    '5EA': '5EA.AX',
    'LGFA': 'LGF-A',
    'YERBU': 'YERB-U.V',
    'DEVO': 'DEVO.L',
    'CRP': 'CRP.TO',
    'BRKA': 'BRK-A',
    'FCIT': 'FCIT.L',
    'GLXY': 'GLXY.TO',
    'NBPE': 'NBPEL.XC',
    'BIOG': 'BIOG.L',
    'BGCG': 'BGCG.L',
    'JZCP': 'JZCP.L',
    'WKOF': 'WKOF.L',
    'NPFU': 'NACQF',
    'MOGA': 'MOG-A',
    'VRNO': 'VRNOF',
    'MWTR': 'MWTR.OL',
    'AVH': 'AVH.AX',
    'AGLX': 'AGLX.OL',
    'PPHC': 'PPHC.L',
    'CCHW': 'CCHW.NE',
    'M7T': 'M7T.AX',
    'SPSY': 'SPSY.L',
    'AYRA': 'AYRWF',
    'NBVA': 'NBVA.V',
    'POLX': 'POLX.L',
    'CWENA': 'CWEN-A',
    'IFOS': 'IFOS.V',
    'ASCU': 'ASCU.TO'
}

EXCLUDE_TICKERS = ["1316", "A900100", "6550", "360", "2257", "1521", "A950160", "3088", "6819", "A950130", "A950140",
                   "6697", "A950200", "4971", "6598", "3664", "RWFC", 'AAWHU', 'BOKU', 'LRBI', 'PBIT', 'FHTF', 'LWEL',
                   'PLTH', 'FFNT', 'STFR', 'MHCUN', 'ACRGAU']

# there are some tickers that are duplicates
# use the below map to adjust the Yahoo ticker based on the company name
NAME_YAHOO_TICKER_MAP = {
    'Green Thumb Industries Inc.': 'GTBIF',
    'Cresco Labs Inc.': 'CRLBF',
    'Aura Minerals Inc.': 'ORA.TO',
    'Argonaut Gold Inc.': 'AR.TO'
}

DATA_AGG_OPERATOR_MAP = {
    'Market Cap': 'sum',
    'Total Debt (LTM)': 'sum',
    'Cash/ST Investments (LTM)': 'sum',
    'Effective Tax Rate - (Ratio) (LTM)': 'median',
    'Beta': ['mean', 'median', 'count']
}

# logger
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

BETA_INPUT_US_STOCK_PATH = get_project_root() / 'data/beta_input_usa_stock.csv'
INDUSTRY_BETA_PATH = get_project_root() / 'data/industry_beta.csv'


def historical_beta(tickers: {str, list}, market_ticker: str = '^GSPC') -> {float, pd.Series}:
    """
    Returns the beta as a float. Beta is calculated as the weighted average of 2-year and 5-year weekly return
    regression betas, with 2-year betas weighted 2/3rds. If the company has only a 2-year beta, it is used.
    This is the method used by Aswath Damodaran
    :param tickers: str or list of str
    :param market_ticker: str
    :return: float or a Series of floats
    """
    start_date_2yr = datetime.now() - timedelta(days=365 * 2)
    start_date_5yr = datetime.now() - timedelta(days=365 * 5)
    # download historical prices of each stocks and the benchmark
    stock_price_df = get_adj_close_price_df(ticker=tickers, start_date=start_date_5yr)
    benchmark_price_df = get_adj_close_price_df(ticker=market_ticker, start_date=start_date_5yr)

    # merge the two DataFrames and calculate the weekly returns
    weekly_returns_df = stock_price_df.join(benchmark_price_df).pct_change(5)

    # calculate the covariance matrix and use the first row to find the covariance and variance of the market
    cov_mtrx_5yr = weekly_returns_df.cov()
    cov_mtrx_2yr = weekly_returns_df.loc[start_date_2yr:, :].cov()
    beta_5yr = cov_mtrx_5yr.iloc[-1, :-1] / cov_mtrx_5yr.iloc[-1, -1]
    beta_2yr = cov_mtrx_2yr.iloc[-1, :-1] / cov_mtrx_2yr.iloc[-1, -1]
    beta = (2 / 3) * beta_2yr + (1 / 3) * beta_5yr
    if isinstance(tickers, str):
        return beta.values[0]
    else:
        beta.rename('Beta', inplace=True)
        return beta


def _calculate_beta_per_stock() -> None:
    """
    Calculates the beta for each ticker that exists in a csv file and added to the original data and stored as a csv
    file
    :return: None
    """
    # load data
    beta_input_df = pd.read_csv(BETA_INPUT_US_STOCK_PATH, index_col=0)

    # remove the previous Beta column if it exist
    if 'Beta' in beta_input_df.columns:
        beta_input_df.drop('Beta', axis=1, inplace=True)

    # add a new column with the corresponding Yahoo ticker
    beta_input_df = _add_yahoo_tickers(df=beta_input_df)

    # create a list of all the tickers to estimate betas for
    ticker_list = list(beta_input_df['Yahoo Ticker'])
    for e in EXCLUDE_TICKERS:
        if e in ticker_list:
            ticker_list.remove(e)

    # calculate the beta for each stock (split the length of the ticker list) and store it as a series
    i = 0
    betas = None
    for sub_list in list_grouper(iterable=ticker_list, n=int(len(ticker_list) / 2)):
        sub_beta = historical_beta(tickers=sub_list)
        if i == 0:
            betas = sub_beta
        else:
            betas = pd.concat([betas, sub_beta], axis=0)
        i += 1

    # merge the data with the calculated betas and save the result in a csv file
    beta_input_df = beta_input_df.merge(betas.to_frame(), left_on='Yahoo Ticker', right_index=True, how='outer')
    beta_input_df.to_csv(BETA_INPUT_US_STOCK_PATH)
    return None


def _add_yahoo_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column named 'Yahoo Ticker' based on an existing 'Ticker' column
    :param df: DataFrame assuming to have a column named 'Ticker'
    :return: DataFrame
    """
    df = df.copy()
    # add a new column with the corresponding Yahoo ticker
    df['Yahoo Ticker'] = df['Ticker'].replace(ADJUSTED_TICKER_MAP)  # convert to Yahoo ticker
    for key, value in NAME_YAHOO_TICKER_MAP.items():  # handle duplicates
        df.loc[df['Name'] == key, 'Yahoo Ticker'] = value
    return df


def _aggregate_beta_per_industry(marginal_tax_rate: {float, None}=0.25) -> None:
    """
    Calculates the unlevered beta (aka asset beta) and cash adjusted unlevered beta and saves the result in a csv file
    :param marginal_tax_rate: float if None the median effective tax_rate rate per industry will be used
    :return: None
    """
    # loads a csv file to a DataFrame containing various data used for beta calculations for each stock including
    # estimated betas
    betas_df = pd.read_csv(BETA_INPUT_US_STOCK_PATH, index_col=0)

    # remove negative betas and larger or equal to 10
    betas_df = betas_df[(0 < betas_df['Beta']) & (betas_df['Beta'] < 10)]

    # group by Industry and aggregate each data according to a specified method defined in DATA_AGG_OPERATOR_MAP
    result = betas_df.groupby('Industry')[list(DATA_AGG_OPERATOR_MAP.keys())].agg(DATA_AGG_OPERATOR_MAP)

    if marginal_tax_rate:
        tax_rate = marginal_tax_rate
    else:
        tax_rate = result['Effective Tax Rate - (Ratio) (LTM)']['median'] / 100

    for agg_beta_method in ['mean', 'median']:
        # unlevered beta (also known as asset beta) using industry mean and median beta estimations
        # Beta / (1 + (1 - tax_rate rate) * (D / E))
        result['Unlevered Beta', agg_beta_method] = result['Beta', agg_beta_method] / (
                    1 + (1 - tax_rate) * result['Total Debt (LTM)', 'sum'] / result['Market Cap', 'sum'])

        # adjust beta according to the share of cahs & marketable securities and firm value (market cap + total debt)
        # Unlevered Beta / (1 - Cash / Firm Value)
        # Firm = Cash & Marketable Securities / (Market Value of Equity + Total Debt)
        result['Cash Adjusted Beta', agg_beta_method] = result['Unlevered Beta', agg_beta_method] / (
                    1 - result['Cash/ST Investments (LTM)', 'sum'] / (
                        result['Market Cap', 'sum'] + result['Total Debt (LTM)', 'sum']))

    # save result in a csv file
    logger.info(f"Aggregate industry betas saved in '{INDUSTRY_BETA_PATH}'")
    result.to_csv(INDUSTRY_BETA_PATH)
    return


def get_industry_beta(industry_weight_map: dict, unlevered: bool = True, beta_agg_method: str = 'mean') -> float:
    """
    Returns the unlevered bottom-up industry beta as the weighted average of industry betas
    :param industry_weight_map: dict industry names as keys (case insensitive) and industry weight as values
    :param beta_agg_method: str
    :param unlevered: bool if True, returns the 'Unlevered Beta' else just the estimated (levered) 'Beta'
    :return: float
    """
    if unlevered:
        beta_col_name = 'Unlevered Beta'
    else:
        beta_col_name = 'Beta'
    return _get_industry_beta(industry_weight_map=industry_weight_map, beta_agg_method=beta_agg_method,
                              beta_col_name=beta_col_name)


def get_cash_adjusted_industry_beta(industry_weight_map: dict, beta_agg_method: str = 'mean') -> float:
    """
    Returns the unlevered cash adjusted bottom-up industry beta as the weighted average of industry betas
    :param industry_weight_map: dict industry names as keys (case insensitive) and industry weight as values
    :param beta_agg_method: str
    :return: float
    """
    return _get_industry_beta(industry_weight_map=industry_weight_map, beta_agg_method=beta_agg_method,
                              beta_col_name='Cash Adjusted Beta')


def _get_industry_beta(industry_weight_map: dict, beta_agg_method: str, beta_col_name: str) -> float:
    """
    Returns the bottom-up industry beta as the weighted average of industry betas.
    :param industry_weight_map: dict industry names as keys (case insensitive) and industry weight as values
    :param beta_agg_method: str
    :param beta_col_name: str
    :return: float
    """
    # make keys case insensitive
    industry_weight_map = {industry.lower(): weight for industry, weight in industry_weight_map.items()}

    # load the data from csv to a DataFrame (header=[0, 1] creates 2 levels columns)
    industry_beta_df = pd.read_csv(INDUSTRY_BETA_PATH, index_col=0, header=[0, 1])
    industry_beta_df.index = industry_beta_df.index.str.lower()

    # check so that all industries exists
    if any(industry not in industry_beta_df.index for industry in industry_weight_map.keys()):
        raise ValueError("Industry in industry_weight_map does not exist in the 'industry_beta.csv' file")

    # map the weights to the industry and remove the rows with nan
    industry_weights = industry_beta_df.index.str.lower().to_frame()['Industry'].map(industry_weight_map).dropna()

    # for the industries with defined weights, look up the beta and calculate the sum product (if one beta in the
    # industry is nan the value will also be nan
    beta = (industry_beta_df[beta_col_name, beta_agg_method].loc[industry_weights.index] * industry_weights).sum(skipna=False)
    return beta


def main():
    """
    Estimates the Betas for all tickers in 'beta_input_usa_stock.csv' and adds a column to the same file
    Then another DataFrame aggregates the Beta and fundamentals based on industry and saves it in 'industry_beta.csv'
    """
    _calculate_beta_per_stock()
    _aggregate_beta_per_industry()


if __name__ == '__main__':
    # main()
    pass
