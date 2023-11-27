"""valuation_template.py"""

import pandas as pd
import numpy as np

from valuation.koyfin_valuation_model import KoyfinStockValuation

# ______________________________________________________________________________________________________________________
# what is the story?
"""
# Growth
Growth above median (est. and hist. are in-line with 75%-ile company). While market environment is currently weak
I expect a more resilient consumer medium term 

Assume 1-5Y of 10% growth that later converges to risk free rate

# Profitability
Margins are not high and it is a sign of low brand power or moat
Est. and hist. are below median peers. As company focuses more on profitability than growth I assume the firm will
reach peer margin as terminal rate

Assume 1-3Y EBIT margin from analyst then converge to terminal EBIT
5Y historical EBIT median
5-10Y converge to peers median margin

# Risk
Almost negative net debt, ICR > 10, good solvency ratios vs other swedish firms leads me to assume 'A' rating
I will also assume that the beta converges to 1.25 at year 5 as the company becomes less of a growth company
No CRP since bulk of Revenues are in northern Europe

# Reinvestments
Historically total reinvestments per Revenue has been around 5% (historical median)

"""

# ______________________________________________________________________________________________________________________
# parameters
company = 'boozt'
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

# discount rate
# set the credit rating and calculate the implied credit spread once and set initial and final beta
valuation.rating = 'A'
credit_spread = valuation.get_credit_spread()
initial_beta = valuation.get_regression_beta(yf_ticker='boozt.st', yf_ticker_benchmark='^OMX')
final_beta = 1.25

# set the two betas and calculate the WACC for each
valuation.credit_spread = credit_spread
valuation.beta = initial_beta
initial_wacc = valuation.get_wacc()
valuation.beta = final_beta
final_wacc = valuation.get_wacc()

# make the initial WACC converge to final WACC over 5 years
valuation.wacc = pd.Series([initial_wacc, None, None, None, final_wacc, None, None, None, None, final_wacc, final_wacc]).interpolate().values.tolist()

# growth
growth_assumption = 0.10
valuation.revenue_growth_fy = pd.Series([company_data['Est Rev CAGR (1Y)'] / 100, growth_assumption, None, None, growth_assumption,
                                         None, None, None, None, risk_free_rate, risk_free_rate]).interpolate().values.tolist()

# profitability
ebit_mrg_median = valuation.koyfin_analyst.get_ebit_margin(capitalize_rnd=False, reporting_type='FQ').iloc[:, 0].median()
ebit_mrg_peer_median = 0.07
ebit_mrg_peer_75pctile = 0.11
ebit_mrg_terminal = ebit_mrg_peer_median
valuation.ebit_mrg = pd.Series([company_data['EBIT Margin - Est Avg (FY1E)'] / 100, company_data['EBIT Margin - Est Avg (FY2E)'] / 100,
                                company_data['EBIT Margin - Est Avg (FY3E)'] / 100, None, ebit_mrg_median, None, None, None, None,
                                ebit_mrg_terminal, ebit_mrg_terminal]).interpolate().values.tolist()

# reinvestments
reinvestment_rate = valuation.koyfin_analyst.get_reinvestments_per_revenue(False, 'ltm').iloc[:, 0].median()
valuation.reinvestments = reinvestment_rate * np.array(valuation.get_revenues())

# ______________________________________________________________________________________________________________________
# monte carlo simulation
sim_num = 10_000
ebit_mrg_bumps = np.random.triangular(-ebit_mrg_terminal, 0.0, ebit_mrg_peer_75pctile - ebit_mrg_terminal + 0.1, size=sim_num)
df = valuation.run_monte_carlo_simulation(return_price_dist=True, plot_result=True, sim_num=sim_num)
print(df)

