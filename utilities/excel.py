"""excel.py"""

from xlsxwriter.utility import xl_rowcol_to_cell

# __________________________________________________________________________________________________________________
# define formats
TITLE_FORMAT_DICT = {'bold': True, 'font_size': 16, 'italic': True, 'underline': True}
MAIN_HEADER_FORMAT_DICT = {'bold': True, 'font_size': 12, 'bg_color': '#DDEBF7', 'top': 1, 'bottom': 1}
FCFF_TABLE_HEADER_FORMAT_DICT = {'bold': True, 'align': 'right'}
FORMULA_FONT_DICT = {'font_color': '#2F75B5', 'italic': True, 'num_format': '#,##0.00'}
PCT_FORMULA_FONT_DICT = {'font_color': '#2F75B5', 'italic': True, "num_format": "0.00%"}
FINAL_FORMULA_FONT_DICT = {'bold': True, 'font_color': '#2F75B5', 'bg_color': '#F2F2F2', 'italic': True, 'num_format': '#,##0.00', 'top': 1, 'bottom': 2}
BOTTOM_BORDER_DICT = {'bottom': 1}
PCT_FORMAT_DICT = {"num_format": "0.00%"}
AMOUNT_FORMAT_DICT = {'num_format': '#,##0.00'}


def name_cell(work_book, sheet_name: str, name: str, row: int, col: int):
    """
    Names a cell in a specific sheet in a workbook
    :param work_book:
    :param sheet_name: str
    :param name: str (converted to
    :param row:
    :param col:
    :return:
    """
    work_book.define_name(name, f'={sheet_name}!{xl_rowcol_to_cell(row, col, row_abs=True, col_abs=True)}')
    return


def write_section_header(work_sheet, text: str, row: int, col: int, col_stop: int = None, header_format=None):
    """
    Writes a header in the first cell and then the format is pasted to the left (up to col_stop) without any text
    :param work_sheet:
    :param text: str
    :param row: int
    :param col: int
    :param col_stop: int (column int)
    :param header_format:
    :return: None
    """
    if col_stop is None:
        col_stop = col
    for col_num in range(col_stop + 1):
        if col_num == 0:
            t = text
        else:
            t = ''
        work_sheet.write(row, col + col_num, t, header_format)
    return


def write_table_from_dict(work_sheet, data: dict, row_start: int, col_start: int, header: list = None,
                          keys_as_row: bool = True, header_format=None, format_map: dict = None, keys_format_map: dict = None) -> None:
    """
    Writes a table in the specified work sheet with the contents in a dict. One can specify if the keys should be listed
    as a row or column and also headers can be specified as well as formats
    :param work_sheet: worksheet object
    :param data: dict
    :param row_start: int
    :param col_start: int
    :param header: list
    :param keys_as_row: bool if True the keys will be listed per column, else per row
    :param header_format: list
    :param format_map: dict with keys included in data dict
    :return: None
    """
    row_offset = 0
    col_offset = 0
    # write the headers (in a column or row)
    if header:
        if keys_as_row:
            work_sheet.write_column(row_start + 1, col_start, header, header_format)
            col_offset += 1
        else:
            work_sheet.write_row(row_start, col_start + 1, header, header_format)
            row_offset += 1

    # write the content in the data dict
    for k, v in data.items():
        if not isinstance(v, list):
            v = [v]
        if format_map:
            cell_format = format_map.get(k, None)
        else:
            cell_format = None
        if keys_format_map:
            key_format = keys_format_map.get(k, None)
        else:
            key_format = None
        work_sheet.write(row_start + row_offset, col_start + col_offset, k, key_format)
        if keys_as_row:
            work_sheet.write_column(row_start + row_offset + 1, col_start + col_offset, v, cell_format)
            col_offset += 1
        else:
            work_sheet.write_row(row_start + row_offset, col_start + col_offset + 1, v, cell_format)
            row_offset += 1
    return


