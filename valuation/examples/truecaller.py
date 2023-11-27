"""valuation_template.py"""
import pandas as pd
import numpy as np

from utilities.yahoo_finance_api import get_adj_close_price_df
from valuation.discount_rate.country_risk_premium import get_country_risk_premium
from valuation.koyfin_valuation_model import KoyfinStockValuation

# what is the story?
"""
Large growth in users ready to be monetized: Revenue growth will come down to earth while margins will improve
* Well positioned in growing market for smartphone users. 2nd highest population with less than 50% smartphone penetration
which is increasing.
* Potential for further growth in Consumer subscription, as fewer than 1 percent of Truecaller’s MAU are 
subscribers
* Limited threat from new players in the Caller ID market because privacy-focused policies for mobile operating systems 
are preventing competitors from rapidly developing the necessary ID database

# Growth
Recent downturn in Revenue growth was short term mostly outside Truecallers control
Most of the drop can be explained in the worse ad market (lower CPM) that seems to pick up especially when economic
conditions improves
* iPhone penetration to increase in India
* high population growth
* less risk for Revenue to be lost to competitors based on moat
* assuming a rebound in the internet ad market 

These points support high growth
< 3Y: Analyst Est.
3Y - 5Y: Extend the estimates using CAGR around 20% since the growth should still be high
5Y - 10Y: Converge to terminal growth rate

# Profitability
Assumes profitability to remain high because of
* Strong moat (limited threat from new players in the Caller ID market because privacy-focused policies for 
mobile operating systems are preventing competitors from rapidly developing the necessary ID database)
* Increase Truecaller for Business as a share of the business mix
* As India becomes richer, more consumer would probably use the subscription services (assuming TrueCaller can
deliver on more features). Altough at the moment the premium subscription does not look very attractive but I believe 
Truecaller will come up with more features making it easier to monetize the huge user base.
3Y: analyst estimate
3Y-5Y converge to 35%
5Y > converge to 30% as the terminal EBIT margin   
Google EBIT margin (LTM) 20Y median = 27% (32%)
Google EBIT margin (LTM) 20Y low = 20% (25%)
Calculating the adjusted EBIT margin when assuming R&D is capitalized (2Y amortized life same as Truecaller)
the EBIT margin improves by around 5%
Terminal EBIT margin = 25%

# Reinvestment
Assume reinvestments are 5% of revenues converging to zero over 10 years

# Discount rate
Truecaller should have a Country Risk Premium since it derives its revenues outside Sweden. Technically, most of the
revenues comes from ads paid by companies not necessarily based in India. However, the eye balls and the end-consumer
are mostly Indian
Even though the firm is generating cash and is highly profitable, it is still considered somewhat of a growth stock 
with a lot of pricing being done for the future. It is also a much more volotile stock than the market index.
A high beta is therefore justified
For debt, Truecaller will borrow money in Sweden so the cost of debt pre-tax_rate should be risk free rate + default spread
Just looking at the ICR, the highest credit rating is motivated. I assume that there should be a higher risk premium due
to the fact that the company is relatively young (IPO 2019) i.e. a higher cost to issue debt. However all solvency 
metrics are in the lower quartile among companies. I will assume AA
"""

# ______________________________________________________________________________________________________________________
# parameters
company = 'trueb'
yf_ticker = 'true-b.st'
implied_equity_risk_premium = 0.0495  # run FundamentalValuation\valuation\discount_rate\implied_equity_risk_premium.py
country_marginal_tax_rate = 0.2
risk_free_rate = 0.0293

# ______________________________________________________________________________________________________________________
# setup the valuation model
valuation = KoyfinStockValuation(implied_equity_risk_premium=implied_equity_risk_premium,
                                 country_marginal_tax_rate=country_marginal_tax_rate,
                                 risk_free_rate=risk_free_rate)
valuation.koyfin_analyst.company = company
valuation.koyfin_analyst.set_financial_reports()
valuation.koyfin_analyst.set_cross_sectional_company_data()

# ______________________________________________________________________________________________________________________
# adjust valuation parameters to fit the story and the specific company
company_data = valuation.koyfin_analyst.get_company_data()

# discount rate: set to high rating and add a (well deserved) country risk premium
valuation.beta = valuation.get_regression_beta(yf_ticker=yf_ticker, yf_ticker_benchmark='^OMX')
valuation.rating = 'AA'
# assuming that Middle East and Africa is split even and 'Other' is United States i.e. zero CRP
geographic_breakdown = {'India': 0.8,
                        'Middle East': 0.05,
                        'Africa': 0.05,
                        'United States': 0.1}
crp = get_country_risk_premium(country_region=geographic_breakdown, relative_vol_adjusted=True)
valuation.country_risk_premium = crp

# adjust growth assumptions
next_fy_growth = company_data['Est Rev CAGR (1Y)'] / 100
fy2_growth = company_data['Revenues - Est YoY % (FY2E)'] / 100
fy3_growth = company_data['Revenues - Est YoY % (FY3E)'] / 100
fy4_growth = company_data['Est Rev CAGR (3Y)'] / 100

# update as of 2 nov 2023
next_fy_growth = -0.0126
fy2_growth = 0.2609
fy3_growth = 0.323

fy5_growth = 0.2

growth_rates = pd.Series([next_fy_growth, fy2_growth, fy3_growth, None, fy5_growth, None, None, None, None,
                          risk_free_rate, risk_free_rate]).interpolate().values.tolist()
valuation.revenue_growth_fy = growth_rates

# adjust profitability assumptions
ebit_mrg_1 = company_data['EBIT Margin - Est Avg (FY1E)'] / 100
ebit_mrg_2 = company_data['EBIT Margin - Est Avg (FY2E)'] / 100
ebit_mrg_3 = company_data['EBIT Margin - Est Avg (FY3E)'] / 100

# update as of 2 nov 2023
ebit_mrg_1 = 0.66719 / 1.75
ebit_mrg_2 = 0.86968 / 2.21
ebit_mrg_3 = 1.19 / 2.92

fy5_ebit_mrg = 0.35
terminal_ebit_mrg = 0.3
ebit_mrg = pd.Series([ebit_mrg_1, ebit_mrg_2, ebit_mrg_3, None, fy5_ebit_mrg, None, None, None, None,
                      terminal_ebit_mrg, terminal_ebit_mrg]).interpolate().values.tolist()
valuation.ebit_mrg = ebit_mrg

# reinvestments
reinvestment_rate = np.linspace(0.05, 0.0, 11)
valuation.reinvestments = reinvestment_rate * np.array(valuation.get_revenues())


# ______________________________________________________________________________________________________________________
# monte carlo simulation
current_price = get_adj_close_price_df(yf_ticker).iloc[-1, 0]
df = valuation.run_monte_carlo_simulation(return_price_dist=True, current_price=current_price)
print(df)

