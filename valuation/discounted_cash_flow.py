"""discounted_cash_flow.py"""

import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell

from utilities.excel import write_table_from_dict
from utilities.excel import write_section_header
from utilities.excel import name_cell
from utilities.excel import TITLE_FORMAT_DICT
from utilities.excel import FORMULA_FONT_DICT
from utilities.excel import MAIN_HEADER_FORMAT_DICT
from utilities.excel import FINAL_FORMULA_FONT_DICT
from utilities.excel import FCFF_TABLE_HEADER_FORMAT_DICT
from utilities.excel import BOTTOM_BORDER_DICT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class IntrinsicValuation:

    def __init__(self, revenues: {list, np.array}, ebit_mrg: {float, list, np.array}, tax_rate: {float, list, np.array},
                 reinvestments: {list, np.array}, wacc: {float, list, np.array}, terminal_growth_rate: float,
                 total_debt: float, cash: float, share_count: {int, float}, next_fy_time: {float, int},
                 minority_interests: float = 0.0, non_operating_assets: float = 0.0, equity_options_value: float = 0.0):
        """
        Performs a FCFF discounted cash flow model
        :param revenues: list, np.array forecasted revenue amounts including at the terminal year
        :param ebit_mrg: float, list, np.array forecasted EBIT margins including at the terminal year
        :param tax_rate: float, list, np.array forecasted tax_rate rates including at the terminal year
        One should start with the effective tax_rate rate (tax_rate expense / taxable income 'EBT')
        If it varies across years use an average like 3yrs
        If effective tax_rate rate < 0 use 0%
        If effective tax_rate rate > 100% use marginal tax_rate rate for the country
        Then make effective tax_rate rate converge to the marginal tax_rate rate
        :param reinvestments: list, np.array forecasted reinvestment amounts including at the terminal year
        :param wacc: float, list, np.array
        :param terminal_growth_rate: float
        :param total_debt: float should include leases
        :param cash: float should include marketable securities
        :param share_count: int or float aggregate all classes of shares into one number excluding shares underlying employee
        stock options
        :param next_fy_time: float or int time to the next FYE measured in years
        :param minority_interests: float this item can be found on the balance sheet
        If the firm owns more than 50% of another firm (or have effective control of it), 100% of the assets, revenue
        and operating income will be reflected in the accounting statements. Thus if the firm owns 60% of another firm,
        we subtract the "value" of the other 40% (otherwise the value is overestimated).
        The book value of this 40% is 'minority interest' on the balance sheet and preferably one should estimate the
        market value using price-to-book ratios for the industry the other firm is operating in
        :param non_operating_assets: float
        market value of non consolidating minority interests and investments whose current and future revenue_est are not
        showing up on the income statement. In koyfin this item is under 'Long Term Investments'
        Estimate market value by looking up the book value and multiply by industry price-to-book ratio
        :param equity_options_value: float
        """
        self.revenues = revenues
        self.ebit_mrg = ebit_mrg
        self.tax_rate = tax_rate
        self.reinvestments = reinvestments
        self.wacc = wacc
        self.terminal_growth_rate = terminal_growth_rate
        self.total_debt = total_debt
        self.cash = cash
        self.share_count = share_count
        self.next_fy_time = next_fy_time
        self.minority_interests = minority_interests
        self.non_operating_assets = non_operating_assets
        self.equity_options_value = equity_options_value

    def get_fcff_calculation_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame with FCFF details
        :return: DataFrame
        """
        result_df = pd.DataFrame({'Revenues': self.revenues, 'EBIT Margin': self.ebit_mrg, 'Tax Rate': self.tax_rate,
                                  'Reinvestment': self.reinvestments})
        result_df.index = result_df.index + 1
        result_df.rename(index={result_df.index[-1]: 'Terminal Year'}, inplace=True)

        result_df['EBIT'] = result_df['Revenues'] * result_df['EBIT Margin']
        result_df['EBIT(1-t)'] = result_df['EBIT'] * (1 - result_df['Tax Rate'])
        result_df['FCFF'] = result_df['EBIT(1-t)'] - result_df['Reinvestment']
        result_df['WACC'] = self.wacc

        result_df['Years to next FYE'] = [self.next_fy_time + i for i in range(len(self.revenues))]
        result_df.loc['Terminal Year', 'Years to next FYE'] = result_df.loc[len(self.revenues) - 1, 'Years to next FYE']
        result_df.loc['Terminal Year', 'FCFF'] = result_df.loc['Terminal Year', 'FCFF'] / (
                    result_df.loc['Terminal Year', 'WACC'] - self.terminal_growth_rate)

        result_df['PV(FCFF)'] = result_df['FCFF'] / ((1 + result_df['WACC']) ** result_df['Years to next FYE'])
        cols = ['Revenues', 'EBIT Margin', 'EBIT', 'Tax Rate', 'EBIT(1-t)', 'Reinvestment', 'FCFF',
                'WACC', 'Years to next FYE', 'PV(FCFF)']
        return result_df[cols]

    def get_pv_terminal_value(self) -> float:
        """
        Returns the estimated present value of the termianl value of the firm i.e. tail value
        :return: float
        """
        # FCFF = Revenues x EBIT Margin x (1 - tax rate) - Reinvestments
        fcff = np.array(self.revenues) * np.array(self.ebit_mrg) * (1 - np.array(self.tax_rate)) - np.array(
            self.reinvestments)
        if isinstance(self.wacc, float):
            wacc = np.array([self.wacc])
        else:
            wacc = np.array(self.wacc)
        terminal_value = fcff[-1] / (wacc[-1] - self.terminal_growth_rate)
        return terminal_value

    def get_pv_operating_assets(self):
        """
        Returns the estimated value of the operating assets (i.e. present value of future FCFF)
        :return: float
        """
        # FCFF = Revenues x EBIT Margin x (1 - tax rate) - Reinvestments
        fcff = np.array(self.revenues) * np.array(self.ebit_mrg) * (1 - np.array(self.tax_rate)) - np.array(
            self.reinvestments)
        if isinstance(self.wacc, float):
            wacc = np.array([self.wacc])
        else:
            wacc = np.array(self.wacc)
        terminal_value = fcff[-1] / (wacc[-1] - self.terminal_growth_rate)
        fcff[-1] = terminal_value
        next_fy = np.array([self.next_fy_time + i for i in range(len(self.revenues))])  # number of years to the next FY
        next_fy[-1] = next_fy[-2]
        pv_fcff = fcff / ((1 + wacc) ** next_fy)
        pv_operating_assets = np.sum(pv_fcff)
        return pv_operating_assets

    def get_estimated_value(self, per_share: bool = True) -> float:
        """
        Returns the estimated value of equity as a float
        :param per_share: bool if True returns the estimated value per share
        :return: float
        """
        pv_operating_assets = self.get_pv_operating_assets()

        equity_value = pv_operating_assets - self.total_debt - self.minority_interests + self.cash + self.non_operating_assets
        common_equity_value = equity_value - self.equity_options_value
        if per_share:
            return common_equity_value / self.share_count
        else:
            return common_equity_value

    def run_monte_carlo_simulation(self, revenue_pct_bumps: np.ndarray, ebit_mrg_bumps: np.ndarray, rate_bumps: np.ndarray,
                                   sim_num: int = 10_000) -> np.ndarray:
        """
        Returns an array with simulated prices based on the specified scenario bumps
        :param revenue_pct_bumps: array
        :param ebit_mrg_bumps: array
        :param rate_bumps: array
        :param sim_num: int
        :return: array
        """
        # base case variables
        revenues_bc = self.revenues
        ebit_mrg_bc = self.ebit_mrg
        wacc_bc = self.wacc
        terminal_growth_rate_bc = self.terminal_growth_rate
        reinvestments_bc = self.reinvestments
        reinvestments_per_revenue = np.array(reinvestments_bc) / np.array(revenues_bc)

        # initialize the result
        result = np.full(shape=sim_num, fill_value=np.nan)

        # loop through each simulation and set the new parameter attributes
        for m in range(sim_num):
            # set parameters for each simulation
            self.revenues = np.array(revenues_bc) * (1 + revenue_pct_bumps[m])
            self.ebit_mrg = ebit_mrg_bc + ebit_mrg_bumps[m]
            # floor the terminal EBIT mrg to zero other wise you will get negative stock prices which is not informative
            self.ebit_mrg[-1] = max(0.0, self.ebit_mrg[-1])
            self.reinvestments = reinvestments_per_revenue * np.array(self.revenues)
            self.wacc = wacc_bc + rate_bumps[m]
            self.terminal_growth_rate = terminal_growth_rate_bc + rate_bumps[m]

            # calculate the price and store the result in an array
            result[m] = self.get_estimated_value()

        # reset variables to base case
        self.revenues = revenues_bc
        self.ebit_mrg = ebit_mrg_bc
        self.reinvestments = reinvestments_bc
        self.wacc = wacc_bc
        self.terminal_growth_rate = terminal_growth_rate_bc

        return result

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
        # sheet_name = 'Valuation'
        # file_path = 'name.xlsx'
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
        fcff_df = self.get_fcff_calculation_df()
        col_headers = [f'FYE{y}' for y in range(1, fcff_df.shape[1] + 1)]
        col_headers.append('Terminal Year')
        fcff_data = fcff_df.to_dict('list')
        data_length = fcff_df.shape[1]

        # replace some value lists with formulas
        fcff_data['EBIT'] = [f'={xl_rowcol_to_cell(table1_row + 1, table1_col + c)} '
                             f'* {xl_rowcol_to_cell(table1_row + 2, table1_col + c)}' for c in
                             range(1, data_length + 2)]
        fcff_data['EBIT(1-t)'] = [f'={xl_rowcol_to_cell(table1_row + 3, table1_col + c)} '
                                  f'* (1 - {xl_rowcol_to_cell(table1_row + 4, table1_col + c)})' for c in
                                  range(1, data_length + 2)]
        fcff_data['FCFF'] = [f'={xl_rowcol_to_cell(table1_row + 5, table1_col + c)} '
                             f'- {xl_rowcol_to_cell(table1_row + 6, table1_col + c)}' for c in
                             range(1, data_length + 2)]
        fcff_data['PV(FCFF)'] = [f'={xl_rowcol_to_cell(table1_row + 7, table1_col + c)} '
                                 f'/ ((1 + {xl_rowcol_to_cell(table1_row + 8, table1_col + c)}) ^ {xl_rowcol_to_cell(table1_row + 9, table1_col + c)})'
                                 for c in range(1, data_length + 1)]
        # terminal value
        fcff_data['PV(FCFF)'].append(f'={xl_rowcol_to_cell(table1_row + 7, table1_col + data_length)} '
                                     f'/ ({xl_rowcol_to_cell(table1_row + 8, table1_col + data_length)} - TERMINAL_GROWTH_RATE)'
                                     f'/ ((1 + {xl_rowcol_to_cell(table1_row + 8, table1_col + data_length)}) ^ {xl_rowcol_to_cell(table1_row + 9, table1_col + data_length)})')

        # user inputs with default data plugged
        assumption_table_content = {
            'Terminal Growth Rate': self.terminal_growth_rate,
            'Total Debt': self.total_debt,
            'Cash': self.cash,
            'Minority Interests': self.minority_interests,
            'Non Operating Assets': self.non_operating_assets,
            'Equity Options': self.equity_options_value,
            'Share Count': self.share_count,
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
        ws.write(0, table1_col, f'Intrinsic Valuation', wb.add_format(TITLE_FORMAT_DICT))
        # main header for cash flow table
        write_section_header(ws, "Free Cash Flow to Firm (FCFF)", row=table1_row - 1, col=table1_col,
                             col_stop=table1_col + data_length, header_format=wb.add_format(MAIN_HEADER_FORMAT_DICT))
        # cash flow table
        write_table_from_dict(work_sheet=ws, data=fcff_data, row_start=table1_row, col_start=table1_col,
                              header=col_headers,
                              keys_as_row=False, header_format=wb.add_format(FCFF_TABLE_HEADER_FORMAT_DICT),
                              format_map={'EBIT': wb.add_format(FORMULA_FONT_DICT),
                                          'EBIT(1-t)': wb.add_format(FORMULA_FONT_DICT),
                                          'FCFF': wb.add_format(FORMULA_FONT_DICT),
                                          'PV(FCFF)': wb.add_format(FINAL_FORMULA_FONT_DICT)},
                              keys_format_map={'PV(FCFF)': wb.add_format(FINAL_FORMULA_FONT_DICT)})

        # main header for assumptions
        assumption_table_row_start = table1_row + len(fcff_data) + rows_between_tables + 2
        write_section_header(ws, "Valuation Assumptions", row=assumption_table_row_start - 1, col=table1_col,
                             col_stop=table1_col + data_length, header_format=wb.add_format(MAIN_HEADER_FORMAT_DICT))

        # assumptions table
        write_table_from_dict(work_sheet=ws, data=assumption_table_content, row_start=assumption_table_row_start,
                              col_start=table1_col, keys_as_row=False,
                              format_map={'Share Count': wb.add_format(BOTTOM_BORDER_DICT)},
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
                             col_stop=table1_col + data_length, header_format=wb.add_format(MAIN_HEADER_FORMAT_DICT))

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
        ws.set_column(table1_col + data_length + 2, table1_col + data_length + 1, 12)

        wb.close()
        return


def get_price_pct_ile(price: float, price_simulation_result: np.ndarray):
    """
    Returns the percentile of a specified price compared to an array of simulated prices
    :param price: float
    :param price_simulation_result: array
    :return: float
    """
    price_simulation_result = price_simulation_result.tolist()
    price_simulation_result.append(price)
    pct_ile = pd.Series(price_simulation_result).rank(pct=True).iloc[-1]
    return pct_ile


def get_price_simulation_distribution(price_simulation_result: np.ndarray, current_price: float = None) -> pd.Series:
    """
    Returns a Series with index labels and 10%, 20%, ..., 90%-ile for the prices
    :param price_simulation_result: array
    :param current_price: float
    :return: Series
    """
    # calculate the percentile price from 10% to 90%-ile
    pct_iles = range(10, 100, 10)
    price_pct_ile = [np.percentile(price_simulation_result, p) for p in pct_iles]
    labels = [f"{p}%-ile" for p in pct_iles]
    # if current price is specified, add it to the distribution table
    if current_price:
        price_pct_ile.append(current_price)
        pct_ile = get_price_pct_ile(price=current_price, price_simulation_result=price_simulation_result)
        labels.append(f'Current price ({round(pct_ile * 100, 1)}%-ile)')
    # store in a series and sort the values
    dist_res = pd.Series(index=labels, data=price_pct_ile).round(2).sort_values()
    return dist_res


def plot_monte_carlo_result(price_simulations: np.ndarray, current_price: float = None, base_case_price: float = None,
                            bearish_case_pct_ile: int = 25, bullish_case_pct_ile: int = 75) -> None:
    """
    Plots the specified simulated prices as well as the bearish/bullish case, base case (if specified) and current price
    (if specified)
    :param price_simulations: array
    :param current_price: float
    :param base_case_price: float
    :param bearish_case_pct_ile: int
    :param bullish_case_pct_ile: int
    :return: None
    """
    # initialize the results to be inputed into a DataFrame
    prices = []
    labels = []
    colors = []

    bearish_px = np.percentile(price_simulations, bearish_case_pct_ile)
    bearish_label = f'Bearish case: {round(bearish_px, 2)} ({bearish_case_pct_ile}%-ile'
    if current_price:
        px_ret = bearish_px / current_price - 1
        sign = '+' if px_ret >= 0 else '-'
        bearish_label += f', {sign}{round(100 * abs(px_ret), 2)}%'
    bearish_label += ')'
    prices.append(bearish_px)
    labels.append(bearish_label)
    colors.append('r')

    bullish_px = np.percentile(price_simulations, bullish_case_pct_ile)
    bullish_label = f'Bullish case: {round(bullish_px, 2)} ({bullish_case_pct_ile}%-ile'
    if current_price:
        px_ret = bullish_px / current_price - 1
        sign = '+' if px_ret >= 0 else '-'
        bullish_label += f', {sign}{round(100 * abs(px_ret), 2)}%'
    bullish_label += ')'
    prices.append(bullish_px)
    labels.append(bullish_label)
    colors.append('g')

    if current_price:
        prices.append(current_price)
        pct_ile = get_price_pct_ile(price=current_price, price_simulation_result=price_simulations)
        labels.append(f'Current price: {round(current_price, 2)} ({round(pct_ile * 100, 1)}%-ile)')
        colors.append('k')

    if base_case_price:
        prices.append(base_case_price)
        l = f'Base case: {round(base_case_price, 2)}'
        if current_price:
            px_ret = base_case_price / current_price - 1
            sign = '+' if px_ret >= 0 else '-'
            l += f' ({sign}{round(100 * abs(px_ret), 2)}%)'
        labels.append(l)
        colors.append('b')
    # store the data in a DataFrame in order to sort the prices (this matters for the label positions)
    df = pd.DataFrame({'prices': prices, 'labels': labels, 'colors': colors})
    df.sort_values(by='prices', inplace=True)

    # plot the simulation result together with the highlighted prices
    plt.hist(price_simulations, bins=200, color='c', alpha=0.65)
    _, max_y = plt.ylim()
    y_pos = 0.95
    for _, row in df.iterrows():
        px = row['prices']
        lbl = row['labels']
        c = row['colors']
        plt.axvline(px, color=c, linestyle='dashed', linewidth=1)
        plt.text(px * 1.01, max_y * y_pos, lbl, color=c, bbox=dict(facecolor='w', edgecolor=c))
        y_pos -= 0.07
    plt.show()

if __name__ == '__main__':
    risk_free_rate = 0.03
    revenues = [100] * 11
    ebit_mrg = 0.1
    tax_rate = 0.3
    reinvestments = 0.02 * np.array(revenues)
    wacc = 0.08
    debt = 1000
    cash = 250
    share_count = 1000

    model = IntrinsicValuation(revenues=revenues,
                               ebit_mrg=ebit_mrg,
                               tax_rate=tax_rate,
                               reinvestments=reinvestments,
                               wacc=wacc,
                               terminal_growth_rate=risk_free_rate,
                               total_debt=debt,
                               cash=cash,
                               share_count=share_count,
                               next_fy_time=1)

    model.valuation_results_to_excel('LOOOOL.xlsx')

