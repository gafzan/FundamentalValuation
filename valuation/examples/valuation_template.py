"""valuation_template.py"""

from valuation.koyfin_valuation_model import KoyfinStockValuation

# ______________________________________________________________________________________________________________________
# what is the story?
"""
...
"""

# ______________________________________________________________________________________________________________________
# parameters
company = None
implied_equity_risk_premium = None  # run FundamentalValuation\valuation\discount_rate\implied_equity_risk_premium.py
country_marginal_tax_rate = None
risk_free_rate = None

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

# ______________________________________________________________________________________________________________________
# monte carlo simulation
df = valuation.run_monte_carlo_simulation(return_price_dist=True)
print(df)

