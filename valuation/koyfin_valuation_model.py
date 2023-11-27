"""koyfin_valuation_model.py"""

import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import date

import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell

from utilities.excel import write_table_from_dict
from utilities.excel import write_section_header
from utilities.excel import name_cell
from utilities.excel import TITLE_FORMAT_DICT
from utilities.excel import FORMULA_FONT_DICT
from utilities.excel import PCT_FORMULA_FONT_DICT
from utilities.excel import MAIN_HEADER_FORMAT_DICT
from utilities.excel import FINAL_FORMULA_FONT_DICT
from utilities.excel import FCFF_TABLE_HEADER_FORMAT_DICT
from utilities.excel import BOTTOM_BORDER_DICT
from utilities.excel import PCT_FORMAT_DICT
from utilities.excel import AMOUNT_FORMAT_DICT

from valuation.discount_rate.company_betas import historical_beta
from valuation.discount_rate.country_risk_premium import get_country_risk_premium
from valuation.discount_rate.cost_of_debt import credit_spread
from valuation.discount_rate.cost_of_debt import synthetic_rating_based_on_icr
from valuation.discounted_cash_flow import IntrinsicValuation
from valuation.discounted_cash_flow import plot_monte_carlo_result
from valuation.discounted_cash_flow import get_price_simulation_distribution

import sys
import logging

sys.path.append(r'C:\Users\gafza\PycharmProjects\Koyfin')
from automation.koyfin_analysis import KoyfinAnalyst

# logger
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)


class KoyfinStockValuation:

    def __init__(self, implied_equity_risk_premium: float, country_marginal_tax_rate: float, risk_free_rate: float):
        self.koyfin_analyst = KoyfinAnalyst()
        self.implied_equity_risk_premium = implied_equity_risk_premium
        self.country_marginal_tax_rate = country_marginal_tax_rate
        self.risk_free_rate = risk_free_rate

        # forward fundamentals
        self.revenues = None
        self.revenue_growth_fy = None
        self.ebit_mrg = None
        self.reinvestments = None
        self.tax_rate = None

        # discount rate
        self.wacc = None
        self.beta = None
        self.country_risk_premium = 0
        self.equity_weight = None
        self.rating = None
        self.credit_spread = None

        # fundamentals
        self.years_to_next_fy = None
        self.number_of_shares = None
        self.total_debt = None
        self.total_cash = None
        self.non_operating_assets = None
        self.minority_interest = 0
        self.equity_options_value = 0

    # this function is called when beta has not been specified
    def get_regression_beta(self, yf_ticker: str = None, yf_ticker_benchmark: str = None, unlevered: bool = True,
                            cash_adjusted: bool = True, floor: float = 0.65, cap: float = 2) -> float:
        """
        Downloads prices from Yahoo finance and calculates the regression beta for a stock. Can adjust the beta for
        leverage and cash
        :param yf_ticker: str if not specified the ticker assigned to the koyfin analyst is used
        :param yf_ticker_benchmark: str if not specified use S&P 500
        :param unlevered: bool
        :param cash_adjusted: bool
        :param floor: float
        :param cap: float
        :return: None
        """
        # download the raw regression beta
        if not yf_ticker:
            yf_ticker = self.koyfin_analyst.ticker
        if yf_ticker_benchmark:
            beta = historical_beta(tickers=yf_ticker, market_ticker=yf_ticker_benchmark)
        else:
            beta = historical_beta(tickers=yf_ticker)
        logger.info(f"Regression beta = {round(beta, 2)}")
        if unlevered:
            # Beta / (1 + (1 - tax_rate rate) * (D / E))
            company_data = self.koyfin_analyst.get_company_data()
            total_debt = company_data['Total Debt (LTM)']
            market_cap = company_data['Market Cap']
            beta = beta / (1 + (1 - self.country_marginal_tax_rate) * total_debt / market_cap)
            logger.info(f"Unlevered beta = {round(beta, 2)}")

        if cash_adjusted:
            # Beta / (1 - Cash / Firm Value)
            company_data = self.koyfin_analyst.get_company_data()
            total_debt = company_data['Total Debt (LTM)']
            market_cap = company_data['Market Cap']
            total_cash = company_data['Cash/ST Investments (LTM)']
            beta = beta / (1 - total_cash / (market_cap + total_debt))
            logger.info(f"Unlevered cash adjusted beta = {round(beta, 2)}")
        return min(cap, max(floor, beta))

    def get_revenue_growth(self) -> list:
        """
        The default model is to use analyst estimates or estimate proxies for year 1-3, then use industry median
        historical growth rates for 5Y and finally use risk free rate as the terminal growth rate
        :return: list
        """
        if self.revenue_growth_fy is None:
            mid_growth_rate = self.koyfin_analyst.get_peer_historical_revenue_growth() / 100
            if self.koyfin_analyst.enough_analyst_coverage():
                logger.info("analyst estimates are used")
                revenue_growth = self.koyfin_analyst.revenue_growth_forecast_analyst(
                    terminal_growth_rate=self.risk_free_rate, extend_analyst_est=False, mid_year_growth=mid_growth_rate)
            else:
                logger.info("proxy analyst estimates are used")
                revenue_growth = self.koyfin_analyst.revenue_growth_forecast_analyst_proxy(
                    terminal_growth_rate=self.risk_free_rate, extend_analyst_est=False, mid_year_growth=mid_growth_rate)
        else:
            revenue_growth = self.revenue_growth_fy
        return revenue_growth

    def get_revenues(self) -> list:
        """
        Returns a list of Revenue amounts where the first Revenue amount is adjusted to not include actual realized
        revenues so far for the fiscal year
        :return: list of floats
        """
        if self.revenues is None:
            revenue_growth = self.get_revenue_growth()
            revenues = self.koyfin_analyst.revenue_forecast(revenue_growth_fy_forecast=revenue_growth)
        else:
            revenues = self.revenues
        return revenues

    def get_ebit_margin(self) -> {list, float}:
        """
        Default model assumes analyst or analyst proxies for year 1-3, then use historical industry median EBIT margins
        as terminal values
        :return: list or float
        """
        if self.ebit_mrg is None:
            terminal_ebit_mrg = self.koyfin_analyst.get_peer_historical_ebit_mrg() / 100
            if self.koyfin_analyst.enough_analyst_coverage():
                logger.info("analyst estimates are used")
                ebit_mrg = self.koyfin_analyst.ebit_mrg_forecast_analyst(terminal_ebit_mrg=terminal_ebit_mrg,
                                                                         extend_analyst_est=False,
                                                                         add_rnd_impact=True)
            else:
                logger.info("proxy analyst estimates are used")
                ebit_mrg = self.koyfin_analyst.ebit_mrg_forecast_analyst_proxy(terminal_ebit_mrg=terminal_ebit_mrg,
                                                                               extend_analyst_est=False,
                                                                               add_rnd_impact=True)
        else:
            ebit_mrg = self.ebit_mrg
        return ebit_mrg

    def get_reinvestments(self, revenues: {np.ndarray, list}=None, use_cap_sales_ratio_estimation: bool = False) -> np.ndarray:
        """
        Returns an array of reinvestments. If not already specified it is estimated based on either capital-to-sales
        ratio or historical reinvestment rates as a share of revenues
        :param revenues: array or list
        :param use_cap_sales_ratio_estimation: bool
        :return: array
        """
        if self.reinvestments is None:
            if revenues is None:
                raise ValueError(
                    "'revenues' needs to be specified when 'reinvestments' attribute is None. If 'reinvestments' is not specified the model will use the capital to sales ratio to "
                    "approximate reinvestments.")
            if use_cap_sales_ratio_estimation:
                logger.debug("estimate reinvestments based on capital-to-sales ratio")
                cap_sales_ratio = self.koyfin_analyst.capital_sales_ratio_forecast()
                delta_rev = np.diff(revenues)
                delta_rev = np.append(delta_rev, revenues[-1] * (1 + self.risk_free_rate) - revenues[-1])
                reinvestments = cap_sales_ratio * delta_rev
            else:
                logger.debug("estimate reinvestments based on historical reinvestment rates as a share of revenues")
                reinvestment_rate = self.koyfin_analyst.reinvestment_rate_forecast()
                reinvestments = np.array(reinvestment_rate) * np.array(revenues)
        else:
            reinvestments = self.reinvestments
        return reinvestments

    def get_tax_rate(self):
        """
        Returns tax rate as float
        :return:
        """
        if self.tax_rate is None:
            logger.debug("estimate tax rate")
            tax_rate = self.koyfin_analyst.tax_rate_forecast(marginal_tax_rate=self.country_marginal_tax_rate)
        else:
            tax_rate = self.tax_rate
        return tax_rate

    def get_wacc(self) -> {float, np.ndarray}:
        """
        Returns WACC as a float or array
        If not specified it is calculated as equity_cost * equity_weight + (1 - marginal taxt) * (1 - equity_weight) * debt_cost
        :return: float or array
        """
        if self.wacc is None:
            logger.debug("wacc not specified")
            cost_of_equity = self.get_cost_of_equity()
            cost_of_debt = self.get_cost_of_debt()
            equity_weight = self.get_equity_weight()
            wacc = cost_of_equity * equity_weight + (1 - self.country_marginal_tax_rate) * (
                        1 - equity_weight) * cost_of_debt
        else:
            wacc = self.wacc
        logger.debug(f"WACC = {round(wacc * 100, 2)}%")
        return wacc

    def get_cost_of_equity(self) -> {float, np.ndarray}:
        """
        Returns cost of equity as a float or array
        :return: float or array
        """
        beta = self.get_beta()
        cost_of_equity = self.risk_free_rate + beta * (self.implied_equity_risk_premium + self.country_risk_premium)
        logger.debug(f'implied equity risk premium: {round(self.implied_equity_risk_premium * 100, 2)}%')
        logger.debug(f'country risk premium: {round(self.country_risk_premium * 100, 2)}%')
        logger.debug(f'cost of equity = {round(cost_of_equity * 100, 2)}%')
        return cost_of_equity

    def get_beta(self) -> {float, np.ndarray}:
        """
        Returns beta as a float or array
        :return: float
        """
        if self.beta:
            beta = self.beta
        else:
            logger.debug("beta not specified so calculate regression beta")
            beta = self.get_regression_beta()
        logger.debug(f'beta = {round(beta, 2)}')
        return beta

    def get_equity_weight(self) -> float:
        """
        Returns equity weight as a float
        :return: float
        """
        if self.equity_weight:
            eq_w = self.equity_weight
        else:
            company_data = self.koyfin_analyst.get_company_data().copy()
            eq_w = company_data['Market Cap'] / (company_data['Market Cap'] + company_data['Total Debt (LTM)'])
        logger.debug(f"equity weight = {round(eq_w * 100, 2)}%")
        return eq_w

    def get_credit_spread(self) -> float:
        """
        Returns credit spread as a float
        :return: float
        """
        if self.credit_spread:
            _credit_spread = self.credit_spread
        else:
            if self.rating:
                rating = self.rating
            else:
                logger.info("calculate synthetic rating based on interest coverage ratio")
                # calculate a synthetic rating from the historical Interest Coverage Ratio over the past 3 years
                icr = self.koyfin_analyst.get_interest_coverage_ratio(report_type='fy', avg_window=3).iloc[-1, 0]
                rating = synthetic_rating_based_on_icr(interest_coverage_ratio=icr)
                logger.info(f"rating: {rating}")
            _credit_spread = credit_spread(rating=rating)
        logger.debug(f"credit spread = {round(_credit_spread * 100, 2)}%")
        return _credit_spread

    def get_cost_of_debt(self) -> float:
        """
        Returns cost of debt as a float
        :return: float
        """
        cost_of_debt = self.risk_free_rate + self.get_credit_spread()
        logger.debug(f"cost of debt = {round(cost_of_debt * 100, 2)}%")
        return cost_of_debt

    def get_next_fy_time(self) -> float:
        """
        Returns a float number of years to next fiscal year end
        By default assumes that next fiscal year end occurs next 31 Dec
        :return: float
        """
        return self.years_to_next_fy if self.years_to_next_fy else (dt(dt.now().year, 12, 31) - dt.now()).days / 365

    def get_valuation_model(self) -> IntrinsicValuation:
        """
        Returns an instance of IntrinsicValuation with populated attributes
        :return: IntrinsicValuation
        """
        # forecasted values
        revenues = self.get_revenues()
        ebit_mrg = self.get_ebit_margin()
        reinvestments = self.get_reinvestments(revenues=revenues)
        tax_rate = self.get_tax_rate()
        wacc = self.get_wacc()

        # company fundamentals
        total_debt = self.total_debt if self.total_debt else \
            self.koyfin_analyst.get_balance_sheet_item('Total Debt', 'ltm').iloc[-1, 0]
        total_cash = self.total_cash if self.total_cash else \
            self.koyfin_analyst.get_balance_sheet_item('Total Cash And Short Term Investments', 'ltm').iloc[-1, 0]
        share_count = self.number_of_shares if self.number_of_shares else \
            self.koyfin_analyst.get_income_statement_item('Basic Weighted Average Shares Outstanding', 'ltm').iloc[
                -1, 0]
        next_fy_time = self.years_to_next_fy if self.years_to_next_fy else (dt(dt.now().year, 12,
                                                                               31) - dt.now()).days / 365
        non_operating_assets = self.non_operating_assets if self.non_operating_assets else \
            self.koyfin_analyst.get_balance_sheet_item('Long Term Investments', 'ltm').fillna(0).iloc[-1, 0]

        # setup a valuation model
        valuation_model = IntrinsicValuation(revenues=revenues, ebit_mrg=ebit_mrg, tax_rate=tax_rate,
                                             reinvestments=reinvestments,
                                             wacc=wacc, terminal_growth_rate=self.risk_free_rate, total_debt=total_debt,
                                             cash=total_cash, share_count=share_count, next_fy_time=next_fy_time,
                                             minority_interests=self.minority_interest,
                                             non_operating_assets=non_operating_assets,
                                             equity_options_value=self.equity_options_value)
        return valuation_model

    def get_estimated_value(self, per_share: bool = True) -> float:
        """
        Returns the estimated value of equity as a float
        :param per_share: bool if True returns the estimated value per share
        :return: float
        """
        valuation_model = self.get_valuation_model()
        return valuation_model.get_estimated_value(per_share=per_share)

    def run_monte_carlo_simulation(self, sim_num: int = 10_000, plot_result: bool = True,
                                   return_price_dist: bool = False,
                                   bearish_pct_ile: int = 25, bullish_pct_ile: int = 75, current_price: float = None,
                                   revenue_pct_bumps: np.ndarray = None, ebit_mrg_bumps: np.ndarray = None,
                                   rate_bumps: np.ndarray = None) -> {None, pd.DataFrame}:
        """
        Runs a monte carlo simulations based on specified input bumps for revenues, EBIT margins and rates. If the bumps
        are not specified default bumps will be used.
        :param sim_num: int
        :param plot_result: bool
        :param return_price_dist: bool if True returns a Series with prices per different percentiles
        :param bearish_pct_ile: int
        :param bullish_pct_ile: int
        :param current_price: float if not specified the price available in a comparable data DataFrame
        :param revenue_pct_bumps: array
        :param ebit_mrg_bumps: array
        :param rate_bumps: array
        :return: Series
        """
        if revenue_pct_bumps is None:
            # simulate revenue bumps base on historical % standard deviation
            revenue_std = (self.koyfin_analyst.get_income_statement_item('total revenues', 'ltm').std() /
                           self.koyfin_analyst.get_income_statement_item('total revenues', 'ltm')).median()
            revenue_pct_bumps = np.random.triangular(-revenue_std, 0, revenue_std, sim_num)

        if ebit_mrg_bumps is None:
            # simulate EBIT margin bumps based on historical standard deviation
            ebit_std = self.koyfin_analyst.get_ebit_margin(False, 'fq').std()
            ebit_mrg_bumps = np.random.triangular(-1.5 * ebit_std, 0, 1.5 * ebit_std, sim_num)

        if rate_bumps is None:
            # simulate rate bumps
            rate_bumps = np.random.normal(0.0, 0.015, sim_num)

        # valuation model used to calculate intrinsic values
        val_model = self.get_valuation_model()
        sim_res = val_model.run_monte_carlo_simulation(revenue_pct_bumps=revenue_pct_bumps,
                                                       ebit_mrg_bumps=ebit_mrg_bumps,
                                                       rate_bumps=rate_bumps, sim_num=sim_num)

        if plot_result:
            if current_price is None:
                current_price = self.koyfin_analyst.get_company_data()['Last Price']
            bc_px = val_model.get_estimated_value()
            plot_monte_carlo_result(price_simulations=sim_res, current_price=current_price, base_case_price=bc_px,
                                    bearish_case_pct_ile=bearish_pct_ile, bullish_case_pct_ile=bullish_pct_ile)

        if return_price_dist:
            if current_price is None:
                current_price = self.koyfin_analyst.get_company_data()['Last Price']
            return get_price_simulation_distribution(price_simulation_result=sim_res, current_price=current_price)

    def reset_data(self) -> None:
        """
        Resets all attributes
        :return: None
        """
        # forward fundamentals
        self.revenues = None
        self.revenue_growth_fy = None
        self.ebit_mrg = None
        self.reinvestments = None
        self.tax_rate = None

        # discount rate
        self.wacc = None
        self.beta = None
        self.country_risk_premium = 0
        self.equity_weight = None
        self.rating = None
        self.credit_spread = None

        # fundamentals
        self.years_to_next_fy = None
        self.number_of_shares = None
        self.total_debt = None
        self.total_cash = None
        self.non_operating_assets = None
        self.minority_interest = 0
        return

    def sanity_check(self):
        """
        Analyses implied metrics of the valuation to act as a sanity check on the result
        :return:
        """
        # TODO finish this
        # CAGR
        revenues = self.get_revenues()
        ltm_revenues = self.koyfin_analyst.get_income_statement_item('total revenues', 'ltm').iloc[-1, 0]
        t = self.get_next_fy_time() + 10
        cagr = (revenues[-2] / ltm_revenues) ** (1 / t) - 1

        # Terminal value / total value
        valuation_model = self.get_valuation_model()
        termial_value = valuation_model.get_pv_terminal_value()
        termial_value_share = termial_value / valuation_model.get_estimated_value(per_share=False)

        # terminal EBIT
        terminal_ebit = self.get_ebit_margin()[-1]

        # ROIC (adds back the reinvestments and compare this number to sector, industry and its historical value


    def valuation_results_to_excel(self, file_path: str, sheet_name: str = 'Valuation'):
        """
        Writes the valuation model as an excel sheet with the relevant formulas and results
        (it looks a bit messy but all it is doing is write 3 different tables (cash flows, inputs & assumptions and result)
        with some cells as formulas)
        :param file_path: str
        :param sheet_name: str
        :return: None
        """
        # __________________________________________________________________________________________________________________
        # setup the excel workbook
        wb = xlsxwriter.Workbook(file_path)
        ws = wb.add_worksheet(sheet_name)
        ws.hide_gridlines(option=2)

        # __________________________________________________________________________________________________________________
        # positions of the tables
        table1_row = 3
        table1_col = 1
        # number of empty rows between the tables including headers
        rows_between_tables = 2

        # __________________________________________________________________________________________________________________
        # data
        model = self.get_valuation_model()
        fcff_df = model.get_fcff_calculation_df()
        col_headers = [f'FYE{y}' for y in range(1, fcff_df.shape[1] + 1)]
        col_headers.append('Terminal Year')
        fcff_df['Beta'] = self.get_beta()
        fcff_df['Revenue Growth'] = None
        fcff_df = fcff_df[['Revenues', 'Revenue Growth', 'EBIT Margin', 'EBIT', 'Tax Rate', 'EBIT(1-t)', 'Reinvestment',
                           'FCFF', 'Beta', 'WACC', 'Years to next FYE', 'PV(FCFF)']]
        fcff_data = fcff_df.to_dict('list')
        data_length = len(col_headers)

        # replace some value lists with formulas
        rev_growth_formulas = [f'={xl_rowcol_to_cell(table1_row + 1, table1_col + c)} ' \
                                   f'/ {xl_rowcol_to_cell(table1_row + 1, table1_col + c - 1)} - 1'
                               for c in range(3, data_length + 1)]
        rev_growth_formulas.insert(0, f'={xl_rowcol_to_cell(table1_row + 1, table1_col + 2)} / ({xl_rowcol_to_cell(table1_row + 1, table1_col + 1)} + REVENUES_FY1) - 1')
        rev_growth_formulas.insert(0, f'=({xl_rowcol_to_cell(table1_row + 1, table1_col + 1)} + REVENUES_FY1) / (REVENUES_FY0) - 1')
        fcff_data['Revenue Growth'] = rev_growth_formulas

        fcff_data['EBIT'] = [f'={xl_rowcol_to_cell(table1_row + 1, table1_col + c)} '
                             f'* {xl_rowcol_to_cell(table1_row + 3, table1_col + c)}' for c in
                             range(1, data_length + 1)]
        fcff_data['EBIT(1-t)'] = [f'={xl_rowcol_to_cell(table1_row + 4, table1_col + c)} '
                                  f'* (1 - {xl_rowcol_to_cell(table1_row + 5, table1_col + c)})' for c in
                                  range(1, data_length + 1)]
        fcff_data['FCFF'] = [f'={xl_rowcol_to_cell(table1_row + 6, table1_col + c)} '
                             f'- {xl_rowcol_to_cell(table1_row + 7, table1_col + c)}' for c in
                             range(1, data_length + 1)]
        fcff_data['WACC'] = [f'=(RISK_FREE_RATE + {xl_rowcol_to_cell(table1_row + 9, table1_col + c)} ' \
                                 f'* (IMPLIED_EQUITY_RISK_PREMIUM + COUNTRY_RISK_PREMIUM)) ' \
                                 f'* EQUITY_WEIGHT + (1 - MARGINAL_TAX_RATE) * (RISK_FREE_RATE + CREDIT_SPREAD) ' \
                                 f'* (1 - EQUITY_WEIGHT)' for c in range(1, data_length + 1)]
        fcff_data['PV(FCFF)'] = [f'={xl_rowcol_to_cell(table1_row + 8, table1_col + c)} '
                                 f'/ ((1 + {xl_rowcol_to_cell(table1_row + 10, table1_col + c)}) ' \
                                     f'^ {xl_rowcol_to_cell(table1_row + 11, table1_col + c)})'
                                 for c in range(1, data_length)]
        # terminal value
        fcff_data['PV(FCFF)'].append(f'={xl_rowcol_to_cell(table1_row + 8, table1_col + data_length)} '
                                     f'/ ({xl_rowcol_to_cell(table1_row + 10, table1_col + data_length)} - TERMINAL_GROWTH_RATE)'
                                     f'/ ((1 + {xl_rowcol_to_cell(table1_row + 10, table1_col + data_length)}) '
                                     f'^ {xl_rowcol_to_cell(table1_row + 11, table1_col + data_length)})')

        # user inputs with default data plugged
        assumption_table_content = {
            'Risk Free Rate': self.risk_free_rate,
            'Terminal Growth Rate': '=RISK_FREE_RATE',
            'Marginal Tax Rate': self.country_marginal_tax_rate,
            'Implied Equity Risk Premium': self.implied_equity_risk_premium,
            'Country Risk Premium': self.country_risk_premium,
            'Equity Weight': self.get_equity_weight(),
            'Credit Spread': self.get_credit_spread(),
            'Revenues FY0': self.koyfin_analyst.get_income_statement_item('total revenues', 'fy').iloc[-1, 0],
            'Revenues FY1': self.koyfin_analyst.get_revenue_ytd_sum(),
            'Total Debt': model.total_debt,
            'Cash': model.cash,
            'Minority Interests': model.minority_interests,
            'Non Operating Assets': model.non_operating_assets,
            'Equity Options': model.equity_options_value,
            'Share Count': model.share_count,
        }

        # result formulas
        result_table_content = {
            'PV Operating Assets': f"=SUM({xl_rowcol_to_cell(table1_row + len(fcff_data), table1_col + 1)}"
            f":{xl_rowcol_to_cell(table1_row + len(fcff_data), table1_col + 1 + data_length)})",
            'Equity Value': '=PV_OPERATING_ASSETS-TOTAL_DEBT+CASH-MINORITY_INTERESTS+NON_OPERATING_ASSETS',
            'Common Equity Value': '=EQUITY_VALUE-EQUITY_OPTIONS',
            'Estimated Value Per Share': '=COMMON_EQUITY_VALUE/SHARE_COUNT'
        }

        # __________________________________________________________________________________________________________________
        # write the excel sheet
        # main title
        sheet_title = f'Intrinsic Valuation {self.koyfin_analyst.company}'
        if self.koyfin_analyst.company != self.koyfin_analyst.ticker:
            sheet_title += f' ({self.koyfin_analyst.ticker})'
        sheet_title += f' - {date.today().strftime("%d %b %Y")}'

        ws.write(0, table1_col, sheet_title, wb.add_format(TITLE_FORMAT_DICT))
        # main header for cash flow table
        write_section_header(ws, "Free Cash Flow to Firm (FCFF)", row=table1_row - 1, col=table1_col,
                             col_stop=table1_col + data_length - 1, header_format=wb.add_format(MAIN_HEADER_FORMAT_DICT))
        # cash flow table
        write_table_from_dict(work_sheet=ws, data=fcff_data, row_start=table1_row, col_start=table1_col,
                              header=col_headers,
                              keys_as_row=False, header_format=wb.add_format(FCFF_TABLE_HEADER_FORMAT_DICT),
                              format_map={'Revenues': wb.add_format(AMOUNT_FORMAT_DICT),
                                          'Revenue Growth': wb.add_format(PCT_FORMAT_DICT),
                                          'EBIT Margin': wb.add_format(PCT_FORMAT_DICT),
                                          'EBIT': wb.add_format(FORMULA_FONT_DICT),
                                          'Tax Rate': wb.add_format(PCT_FORMAT_DICT),
                                          'EBIT(1-t)': wb.add_format(FORMULA_FONT_DICT),
                                          'Reinvestment': wb.add_format(AMOUNT_FORMAT_DICT),
                                          'FCFF': wb.add_format(FORMULA_FONT_DICT),
                                          'Beta': wb.add_format(AMOUNT_FORMAT_DICT),
                                          'WACC': wb.add_format(PCT_FORMULA_FONT_DICT),
                                          'Years to next FYE': wb.add_format(AMOUNT_FORMAT_DICT),
                                          'PV(FCFF)': wb.add_format(FINAL_FORMULA_FONT_DICT)},
                              keys_format_map={'PV(FCFF)': wb.add_format(FINAL_FORMULA_FONT_DICT)})

        # main header for assumptions
        assumption_table_row_start = table1_row + len(fcff_data) + rows_between_tables + 2
        write_section_header(ws, "Valuation Assumptions", row=assumption_table_row_start - 1, col=table1_col,
                             col_stop=table1_col + data_length - 1, header_format=wb.add_format(MAIN_HEADER_FORMAT_DICT))

        # assumptions table
        pct_ass_format_map = {
            var: wb.add_format(PCT_FORMAT_DICT)
            for var in ['Risk Free Rate', 'Terminal Growth Rate', 'Marginal Tax Rate', 'Implied Equity Risk Premium',
                        'Country Risk Premium', 'Equity Weight', 'Credit Spread']
        }
        amt_ass_format_map = {
            var: wb.add_format(AMOUNT_FORMAT_DICT)
            for var in ['Total Debt', 'Cash', 'Minority Interests', 'Non Operating Assets', 'Equity Options',
                        'Revenues FY0', 'Revenues FY1']
        }
        ass_format_map = pct_ass_format_map.copy()
        ass_format_map.update(amt_ass_format_map)
        ass_format_map.update({'Share Count': wb.add_format(BOTTOM_BORDER_DICT)})
        write_table_from_dict(work_sheet=ws, data=assumption_table_content, row_start=assumption_table_row_start,
                              col_start=table1_col, keys_as_row=False,
                              format_map=ass_format_map,
                              keys_format_map={'Share Count': wb.add_format(BOTTOM_BORDER_DICT)})

        # name all the cells in the assumptions table
        for row, name in enumerate(list(assumption_table_content.keys())):
            name = name.replace(' ', '_').upper()
            name_cell(work_book=wb, sheet_name=sheet_name, name=name, row=assumption_table_row_start + row,
                      col=table1_col + 1)

        # main header for results
        results_table_row_start = table1_row + len(fcff_data) + len(assumption_table_content) + rows_between_tables + 5
        write_section_header(ws, "Valuation Results", row=results_table_row_start - 1,
                             col=table1_col,
                             col_stop=table1_col + data_length - 1, header_format=wb.add_format(MAIN_HEADER_FORMAT_DICT))

        # results table
        write_table_from_dict(work_sheet=ws, data=result_table_content, row_start=results_table_row_start,
                              col_start=table1_col, keys_as_row=False,
                              format_map={'Estimated Value Per Share': wb.add_format(FINAL_FORMULA_FONT_DICT),
                                          'PV Operating Assets': wb.add_format(FORMULA_FONT_DICT),
                                          'Equity Value': wb.add_format(FORMULA_FONT_DICT),
                                          'Common Equity Value': wb.add_format(FORMULA_FONT_DICT)},
                              keys_format_map={'Estimated Value Per Share': wb.add_format(FINAL_FORMULA_FONT_DICT)})

        # name all the cells in the results table
        for row, name in enumerate(list(result_table_content.keys())):
            name = name.replace(' ', '_').upper()
            name_cell(work_book=wb, sheet_name=sheet_name, name=name, row=results_table_row_start + row,
                      col=table1_col + 1)

        # increase the width for some columns
        ws.set_column(table1_col, table1_col, 23)
        ws.set_column(table1_col + 1, table1_col + data_length - 1, 11)
        ws.set_column(table1_col + data_length, table1_col + data_length, 13)

        wb.close()
        return


if __name__ == '__main__':
    # setup valuation model
    valuation = KoyfinStockValuation(implied_equity_risk_premium=0.048, country_marginal_tax_rate=0.21,
                                     risk_free_rate=0.03)
    # define the company to be analysed
    valuation.koyfin_analyst.company = 'nibeb'
    valuation.credit_spread = 0.02
    # valuation.koyfin_analyst.download_financial_reports()
    valuation.koyfin_analyst.set_financial_reports()
    valuation.koyfin_analyst.set_cross_sectional_company_data()

    # adjust valuation inputs
    valuation.beta = valuation.get_regression_beta(yf_ticker='nibe-b.st', yf_ticker_benchmark='^OMX')
    # valuation.rating = 'BBB-'
    # valuation.revenue_growth_fy = valuation.koyfin_analyst.revenue_growth_forecast_analyst(terminal_growth_rate=0.03, extend_analyst_est=False, mid_year_growth=None)

    # monte carlo simulation

    # save to excel
    file_path = r'C:\Users\gafza\OneDrive\Dokument\Finance\valuation\company_valuations\{0}_{1}.xlsx'.format(valuation.koyfin_analyst.company.replace(' ', '_').lower(), date.today().strftime("%Y%m%d"))
    valuation.valuation_results_to_excel(file_path=file_path)


