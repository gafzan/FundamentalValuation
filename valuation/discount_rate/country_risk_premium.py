"""country_risk_premium.py
Run main to calculate country risk premiums for countries and regions. Input data (CDS, ratings and GDP levels) are
stored in csv files.
CDS: country_cds.csv values taken from Professor Aswath Damodaran website (https://pages.stern.nyu.edu/~adamodar/)
Ratings: country_rating.csv (https://tradingeconomics.com/country-list/rating)
GDP: country_gdp.csv (https://tradingeconomics.com/country-list/gdp)
"""

import pandas as pd
import numpy as np
import unicodedata

from utilities.general import get_project_root
from utilities.yahoo_finance_api import get_adj_close_price_df

CDS_COUNTRY_NAME_CONVERTER = {'Korea': 'South Korea', 'Guatamela': 'Guatemala'}
SP_MOODYS_RATING_MAP = {
    "A": "A2",
    "A-": "A3",
    "A+": "A1",
    "AA": "Aa2",
    "AA-": "Aa3",
    "AA+": "Aa1",
    "AAA": "Aaa",
    "B": "B2",
    "B-": "B3",
    "B+": "B1",
    "BB": "Ba2",
    "BB-": "Ba3",
    "BB+": "Ba1",
    "BBB": "Baa2",
    "BBB-": "Baa3",
    "BBB+": "Baa1",
    "C": "C2",
    "C-": "C3",
    "C+": "C1",
    "CC": "Ca2",
    "CC-": "Ca3",
    "CC+": "Ca1",
    "CCC": "Caa2",
    "CCC-": "Caa3",
    "CCC+": "Caa1",
    'D': 'Caa1',
    'SD': 'Caa1'
}

REGION_COUNTRY_MAP = {
    'Africa': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Cameroon', 'Cape Verde', 'Congo', 'Egypt',
               'Ethiopia', 'Gabon', 'Ghana', 'Ivory Coast', 'Kenya', 'Lesotho', 'Madagascar', 'Mali', 'Mauritius',
               'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Republic of the Congo', 'Rwanda', 'Senegal',
               'Seychelles', 'South Africa', 'Swaziland', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia'],
    'Asia': ['Bangladesh', 'Cambodia', 'China', 'Fiji', 'Hong Kong', 'India', 'Indonesia', 'Japan', 'Laos', 'Macau',
             'Malaysia', 'Maldives', 'Mongolia', 'Pakistan', 'Papua New Guinea', 'Philippines', 'Singapore',
             'Solomon Islands', 'South Korea', 'Sri Lanka', 'Taiwan', 'Thailand', 'Vietnam'],
    'Australia & New Zealand': ['Australia', 'New Zealand'],
    'Caribbean': ['Aruba', 'Bahamas', 'Barbados', 'Bermuda', 'Cayman Islands', 'Cuba', 'Dominican Republic', 'Grenada',
                  'Jamaica', 'Montserrat', 'Puerto Rico', 'St Vincent and the Grenadines', 'Trinidad and Tobago'],
    'Central and South America': ['Argentina', 'Belize', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Costa Rica',
                                  'Ecuador', 'El Salvador', 'Guatemala', 'Honduras', 'Mexico', 'Nicaragua', 'Panama',
                                  'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela'],
    'Eastern Europe & Russia': ['Albania', 'Armenia', 'Azerbaijan', 'Belarus', 'Bosnia and Herzegovina', 'Bulgaria',
                                'Croatia', 'Czech Republic', 'Estonia', 'Georgia', 'Hungary', 'Kazakhstan',
                                'Kyrgyzstan', 'Latvia', 'Lithuania', 'Macedonia', 'Moldova', 'Montenegro', 'Poland',
                                'Romania', 'Russia', 'Serbia', 'Slovakia', 'Slovenia', 'Tajikistan', 'Turkmenistan',
                                'Ukraine', 'Uzbekistan'],
    'Middle East': ['Abu Dhabi', 'Bahrain', 'Dubai', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Oman', 'Qatar',
                    'Saudi Arabia', 'United Arab Emirates'],
    'North America': ['Canada', 'United States'],
    'Western Europe': ['Andorra', 'Austria', 'Belgium', 'Cyprus', 'Denmark', 'Finland', 'France', 'Germany', 'Greece',
                       'Iceland', 'Ireland', 'Isle of Man', 'Italy', 'Liechtenstein', 'Luxembourg', 'Malta',
                       'Netherlands', 'Norway', 'Portugal', 'San Marino', 'Spain', 'Sweden', 'Switzerland', 'Turkey',
                       'United Kingdom'],
}

COUNTRY_RISK_PREMIUM_PATH = get_project_root() / 'data/country_data/country_risk_premium.csv'
COUNTRY_RATING_PATH = get_project_root() / 'data/country_data/country_rating.csv'
COUNTRY_CDS_PATH = get_project_root() / 'data/country_data/country_cds.csv'
COUNTRY_GDP_PATH = get_project_root() / 'data/country_data/country_gdp.csv'


def get_country_risk_premium(country_region: {str, dict}, relative_vol_adjusted: bool) -> float:
    """
    Returns the country risk premium as a float. Inputs can be a name of a country/region or a dictionary with
    countries/regions and weights
    :param country_region: str or dict
    :param relative_vol_adjusted: bool (if True the calculated CRP will be adjusted according to the relative volatility
    of EM stocks and bonds
    :return: float
    """
    # load the data from csv to a DataFrame
    crp_df = pd.read_csv(COUNTRY_RISK_PREMIUM_PATH, index_col='Country')
    crp_df.index = crp_df.index.str.lower()

    # make keys case insensitive and covert str input to dict
    ctry_rgns_weight_map = _adjust_crp_inputs(country_region=country_region, crp_df=crp_df)
    crp_column_df = _get_specified_crp_columns(relative_vol_adjusted=relative_vol_adjusted, crp_df=crp_df)

    # map the weights to the country/region and calculate a sum product of the weight and CRP
    ctry_rgns_weights = crp_df.index.str.lower().to_frame()['Country'].map(ctry_rgns_weight_map).dropna()

    # for the industries with defined weights, look up the beta and calculate the sum product (if one beta in the
    # industry is nan the value will also be nan
    crp_series = crp_column_df.loc[ctry_rgns_weights.index]
    countries_with_na = crp_series[crp_series.isna()].index.str.title().tolist()
    if len(countries_with_na):
        raise ValueError("Following countries has NA country risk premium: '%s'" % "' ,'".join(countries_with_na))
    crp = (crp_column_df.loc[ctry_rgns_weights.index] * ctry_rgns_weights).sum(skipna=False)
    return crp


def _adjust_crp_inputs(country_region: {str, dict}, crp_df: pd.DataFrame) -> dict:
    """
    Returns a dictionary with the keys (name of country) in lower case and the weights as values. Checks so weights sum
    to 100%  and converts a str to a dictionary as {name of country: 1.0}. Also checks so that all provided countries
    and regions exists in the country risk premium DataFrame
    :param country_region: str or dict
    :param crp_df: DataFrame
    :return: dict
    """
    if isinstance(country_region, str):
        country_region = {country_region: 1.0}
    country_region = {country_region_name.lower(): weight for country_region_name, weight in country_region.items()}

    # check so that all countries or regions exists in the loaded data
    if any(ctry_rgn not in crp_df.index for ctry_rgn in country_region.keys()):
        raise ValueError(f"Country or region in country_region does not exist in the '{COUNTRY_RISK_PREMIUM_PATH}' file")

    # check weight sum
    w_sum = round(sum(country_region.values()), 2)
    if w_sum != 1.0:
        raise ValueError(f"Sum of country/region weights not equal to 100% (sum: {w_sum * 100}%)")
    return country_region


def _get_specified_crp_columns(relative_vol_adjusted: bool, crp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one column being with or without relative volatility
    adjustment CRP or Rating based CRP depending on data availability
    :param relative_vol_adjusted: bool
    :param crp_df: DataFrame
    :return: DataFrame
    """
    def _assign_non_nan_value(row):
        """
        Helper function for _get_specified_crp_columns
        """
        if relative_vol_adjusted:
            extra_str = 'Adj '
        else:
            extra_str = ''

        if not np.isnan(row[extra_str + 'CRP']):
            return row[extra_str + 'CRP']
        else:
            return row[extra_str + 'Rating CRP']

    return crp_df.apply(_assign_non_nan_value, axis=1)


def get_country_rating_df() -> pd.DataFrame:
    """
    Stores and returns the country rating in a DataFrame and creates a new column with adjusted country rating using
    S&P rating if Moody's does not exist
    :return: DataFrame
    """
    # store the country rating in a DataFrame and create a new column with adjusted country rating using S&P rating if
    # Moody's does not exist
    rating_df = pd.read_csv(COUNTRY_RATING_PATH)
    rating_df['Rating'] = rating_df.apply(_adjusted_country_rating, axis=1)
    return rating_df


def get_country_cds_df() -> pd.DataFrame:
    """
    Returns a DataFrame with a Country column and CDS column
    :return:
    """
    cds_df = pd.read_csv(COUNTRY_CDS_PATH)
    cds_df.columns = ['Country', 'CDS']
    cds_df['Country'] = cds_df['Country'].replace(CDS_COUNTRY_NAME_CONVERTER)
    return cds_df


def _adjusted_country_rating(row):
    """
    Returns the Moody's rating if it is available, else takes the S&P rating and convert it into Moody's format, else
    return NaN
    :param row: series
    :return:
    """
    if pd.isnull(row["Moody's"]) or row["Moody's"] == 'NaN':
        if not pd.isnull(row["S&P"]) or row["S&P"] == 'NaN':
            # unicodedata.normalize replaces all compatibility characters with their equivalents (\xa0 to ' ')
            return SP_MOODYS_RATING_MAP[unicodedata.normalize('NFKC', row["S&P"]).replace(' ', '')]
        else:
            return np.nan
    else:
        return unicodedata.normalize('NFKC', row["Moody's"]).replace(' ', '')


def save_country_risk_premium_csv(rel_vol_ratio: float) -> None:
    """
    Saves a DataFrame with country risk premiums adjusted and non-adjusted for the relative volatility between
    emerging equity and bonds in a csv file.
    :param rel_vol_ratio: float
    :return: None
    """
    # gather the data to use: country sovereign rating and credit default swap levels and merge the results
    rating_df = get_country_rating_df()
    cds_df = get_country_cds_df()
    total_df = rating_df.merge(cds_df, how='outer', on='Country')

    # average CDS per country rating
    avg_cds_per_rating_df = total_df.groupby(by='Rating')['CDS'].mean()

    # add a column with the estimated CDS level based on the country's credit rating
    total_df['Rating CDS'] = total_df['Rating'].replace(avg_cds_per_rating_df.to_dict())

    # because the developed market equity risk premium is based on the United States, the country risk premium should be
    # the credit risk in excess of the United States
    # Country Risk Premium = Max(0, Country CDS - United States CDS)
    total_df['CRP'] = np.maximum(total_df['CDS'] - total_df[total_df['Country'] == 'United States']['CDS'].values, 0)
    total_df['Rating CRP'] = np.maximum(total_df['Rating CDS'] - total_df[total_df['Country'] == 'United States']['CDS'].values, 0)

    # apply a scaling factor to the country risk premium to reflect the fact that equity is more volatile than bonds
    total_df['Adj CRP'] = rel_vol_ratio * total_df['CRP']
    total_df['Adj Rating CRP'] = rel_vol_ratio * total_df['Rating CRP']

    # add region CRP
    total_df = _add_region_country_risk_premium(crp_df=total_df)
    total_df.reset_index(drop=True, inplace=True)  # the added rows all have index 0 so reset the index

    # save the result in a csv file
    total_df.to_csv(COUNTRY_RISK_PREMIUM_PATH)
    return


def em_equity_bond_relative_vol_ratio():
    """
    Returns the relative volatility ratio between emerging equity and bond ETF
    iShares JP Morgan USD Emerging Markets Bond ETF (EMB)
    SPDR Portfolio Emerging Markets ETF (SPEM)
    :return: float
    """
    em_bond_vol_df = get_adj_close_price_df('EMB')['EMB'].pct_change().rolling(252).std() * np.sqrt(252)
    em_equity_vol_df = get_adj_close_price_df('SPEM')['SPEM'].pct_change().rolling(252).std() * np.sqrt(252)
    avg_5yr_bond_vol = em_bond_vol_df.iloc[::-252].iloc[:5].mean()
    avg_5yr_equity_vol = em_equity_vol_df.iloc[::-252].iloc[:5].mean()
    return avg_5yr_equity_vol / avg_5yr_bond_vol


def _add_region_country_risk_premium(crp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rows to the specified DataFrame with region risk premiums as GDP weighted sum of country risk premiums in that
    region
    :param crp_df: DataFrame
    :return: DataFrame
    """
    # retrieve GDP data and merge it with the country risk premiums
    gdp_df = pd.read_csv(COUNTRY_GDP_PATH)
    crp_df = crp_df.merge(gdp_df[['Country', 'Last']], left_on='Country', right_on='Country', how='left')
    crp_df.rename({'Last': 'GDP'}, axis=1, inplace=True)

    # for each region add a row in the crp DataFrame with the GDP weighted CRPs
    for region, countries_per_region in REGION_COUNTRY_MAP.items():
        countries_per_region_df = crp_df[crp_df.Country.isin(countries_per_region)].copy()
        gdp_region_sum = countries_per_region_df['GDP'].sum()

        # sum the product of the CRPs and the countries weight according to GDP
        # if there are no numeric values, return nan instead of 0 using .sum(min_count=1)
        region_crp = countries_per_region_df[['CRP', 'Rating CRP', 'Adj CRP', 'Adj Rating CRP']].mul(countries_per_region_df['GDP'] / gdp_region_sum, axis=0).sum(min_count=1)
        region_crp['Country'] = region
        region_crp['GDP'] = countries_per_region_df['GDP'].sum(min_count=1)
        crp_df = pd.concat([crp_df, region_crp.to_frame().T])  # adds a new row in the DataFrame

    # the added rows all have index 0 so reset the index
    crp_df.reset_index(drop=True, inplace=True)
    return crp_df


def main():
    # calculate country risk premiums using country CDS and rating data
    rel_vol_ratio = em_equity_bond_relative_vol_ratio()
    rel_vol_ratio = 1.42
    save_country_risk_premium_csv(rel_vol_ratio=rel_vol_ratio)


if __name__ == '__main__':
    main()







