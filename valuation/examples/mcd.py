"""mcd.py"""
import pandas as pd

from valuation.koyfin_valuation_model import KoyfinStockValuation
from valuation.discount_rate.country_risk_premium import get_country_risk_premium
from utilities.fred import get_us_10y_cms

# step 1: what is the story
"""
# MCD is not a growth story...
It has experienced relatively high growth recently due to being an inflation winner but since I 
expect that inflation has topped, I expect growth to go back to historical levels and to quickly 
converge to steady state. 
The Revenue stream that MCD receives from the franchise model are predictable and stable.
There are pushes for digitization and delivery but I don't expect this to be 
revolutionising for the business

... it is however a profitability/efficiency story that is expected to remain strong
EBIT margin is higher than industry peers due to MCD's franchise model. MCD can use this model going forward because of
the powerful brand. MCD leases property from landowner and then sub-leases properties to franchisee. 80% of MCD 
locations are franchises. The franchisee are motivated by the impressive margins acting almost like a guaranteed money 
machine so they are willing to lease the properties at a big markup (in favour of MCD). The franchisee is responsible 
for SG&A costs, COGS and paying rents. 
The margins MCD makes on the franchise locations are a lot higher than the company owned ones since the operating costs 
is a lot higher for the latter.

MCD has a goal of franchising more and more of their restaurants up to 95%.
All this motivates that EBIT margins remains high especially vs. competition

Inputs for base case
# Revenues: As a base case I will use analysts Revenue estimates for the coming 3 years and after that 
growth by the risk free rate. Converge to risk-free rate after 3 years

# EBIT Margin: use analyst forecast then converge to 10Y median as steady state

# Reinvestment: assume 2% reinvestment as a % of Revenues that converges to industry median around 1.6%. 
This is in-line with historical values
"""

# parameters
implied_equity_risk_premium = 0.0475
country_marginal_tax_rate = 0.25
risk_free_rate = get_us_10y_cms().iloc[-1]

# setup the valuation model
mcd_valuation = KoyfinStockValuation(implied_equity_risk_premium=implied_equity_risk_premium,
                                     country_marginal_tax_rate=country_marginal_tax_rate,
                                     risk_free_rate=risk_free_rate)
mcd_valuation.koyfin_analyst.company = 'mcd'
mcd_valuation.koyfin_analyst.set_financial_reports()
mcd_valuation.koyfin_analyst.set_cross_sectional_company_data()

# adjust valuation parameters to fit the story and the specific company

mcd_valuation.rating = 'BBB+'
# top 20 countries based on the share of restaurants in each country
# (MCD does not give a revenue breakdown by country)
# Russia has NA country risk premium so I will use Iraq (baddie country)
country_risk_map = {"United States": 0.404800096849369, "China": 0.105928997306377, "Japan": 0.0877697406252838,
                    "Germany": 0.0464876971035986, "France": 0.045398141702733,
                    "Canada": 0.0441269937350565, "United Kingdom": 0.043007172906389,
                    "Brazil": 0.0312036560636785, "Australia": 0.0296903846735874,
                    "Iraq": 0.0257256136315487, "Philippines": 0.0198238552101934,
                    "Italy": 0.0193698737931661, "Spain": 0.0166459852910021, "Poland": 0.0150116521897037,
                    "South Korea": 0.0135286462274144, "Taiwan": 0.0124996216821525,
                    "Mexico": 0.0121667019763324, "Saudi Arabia": 0.00920069005175389,
                    "India": 0.0090796283405466, "Malaysia": 0.0085348506401138}
mcd_valuation.country_risk_premium = get_country_risk_premium(country_region=country_risk_map, relative_vol_adjusted=True)

# data
mcd_data = mcd_valuation.koyfin_analyst.get_company_data()

# adjust revenues to not extend the analyst expectations to 5Y but start to converge to terminal rate
revenue_fy1e = mcd_data['Est Rev CAGR (1Y)'] / 100
revenue_fy2e = mcd_data['Revenues - Est YoY % (FY2E)'] / 100
revenue_fy3e = mcd_data['Revenues - Est YoY % (FY3E)'] / 100
mcd_valuation.revenue_growth_fy = pd.Series([revenue_fy1e, revenue_fy2e, revenue_fy3e, None, None, None, None, None,
                                             None, mcd_valuation.risk_free_rate, mcd_valuation.risk_free_rate]).interpolate().values.tolist()

# adjust ebit margin to start converge to terminal rate after 3Y
ebit_mrg_1 = mcd_data['EBIT Margin - Est Avg (FY1E)'] / 100
ebit_mrg_2 = mcd_data['EBIT Margin - Est Avg (FY2E)'] / 100
ebit_mrg_3 = mcd_data['EBIT Margin - Est Avg (FY3E)'] / 100
terminal_ebit = 0.3
mcd_valuation.ebit_mrg = pd.Series([ebit_mrg_1, ebit_mrg_2, ebit_mrg_3, None, None, None, None, None, None,
                                    terminal_ebit, terminal_ebit]).interpolate().values.tolist()

# monte carlo simulation
df = mcd_valuation.run_monte_carlo_simulation(return_price_dist=True)
print(df)

