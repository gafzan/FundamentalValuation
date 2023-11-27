"""cost_of_debt.py"""

from utilities.fred import get_us_corp_option_adjusted_spread


def credit_spread(rating: str, as_of_date: str = None)->float:
    """
    Returns the credit spread for a given rating and date
    :param rating: str (S&P rating)
    :param as_of_date:  str
    :return: float
    """
    credit_spread_per_rating_df = get_us_corp_option_adjusted_spread(interpolate_additional_ratings=True)
    credit_spread_per_rating_df.fillna(method='ffill', inplace=True)
    if as_of_date:
        return credit_spread_per_rating_df.iloc[as_of_date, rating]
    else:
        # return the latest available data
        return credit_spread_per_rating_df.iloc[-1].loc[rating]


def synthetic_rating_based_on_icr(interest_coverage_ratio: float)->str:
    """
    Returns a string with the rating based on the specified interest coverage ratio
    :param interest_coverage_ratio: float
    :return: str
    """
    if interest_coverage_ratio > 50:
        return 'AAA'
    elif 20 < interest_coverage_ratio <= 50:
        return 'AA'
    elif 10 < interest_coverage_ratio <= 20:
        return 'A'
    elif 5 < interest_coverage_ratio <= 10:
        return 'BBB'
    elif 3 < interest_coverage_ratio <= 5:
        return 'BB'
    elif interest_coverage_ratio <= 3:
        return 'B'

