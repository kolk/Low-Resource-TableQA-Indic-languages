new_raw_tables = {}
def process_table_headers(rows):
    table_headers = []
    num_rows_in_header = 1
    header_rowspan = {}
    header_indx = 0
    for cell in rows[0].find_all(['th', 'td']):
        if cell.has_attr('colspan'):
            spans_present = True
            table_headers += [cell.text.strip()] * int(cell.get('colspan'))
            # include rowspan info colspan number of times
            if cell.has_attr('rowspan'):
                span_value = int(cell.get('rowspan'))
            else:
                span_value = 1
            for t in range(int(cell.get('colspan'))):
                header_rowspan[t + header_indx] = span_value
            header_indx += int(cell.get('colspan'))
            # header_rowspan[header_indx] = 1
        else:
            table_headers.append(cell.text.strip())
            # include rowspan info 1 time
            if cell.has_attr('rowspan'):
                spans_present = True
                span_value = int(cell.get('rowspan'))
            else:
                span_value = 1
            header_rowspan[header_indx] = span_value
            header_indx += 1
    # ----------------------------------------------------------------
    num_rowspans_in_header = max(header_rowspan.values())
    header_row_index = 0
    while num_rowspans_in_header > 1:
        header_row_index += 1
        # print(f'header spans {url}')
        num_rows_in_header += 1
        indices = [k for k, v in header_rowspan.items() if v == 1]

        # extract the next header row from [th,td]
        next_header_row = []
        for cell in rows[header_row_index].find_all(['th', 'td']):
            if cell.has_attr('colspan'):
                next_header_row += [cell.text.strip()] * int(cell.get('colspan'))
            else:
                next_header_row.append(cell.text.strip())
        assert (
        len(indices) == len(next_header_row), "length of next header row not equal to length of rowspan with value 1")

        # concatenate the values of new header row to the original header
        for i, indx in enumerate(indices):
            table_headers[indx] += (" (" + str(next_header_row[i].strip()) + ")")

        # lower the already processed header_rowspan value by 1 and include rowspan from this <tr>
        for k, v in header_rowspan.items():
            if v > 1:
                header_rowspan[k] -= 1  # decrement rowspan by 1 as it is already processed

        # update rowspan dictionary with rowspan info from current <tr>
        t = 0
        for cell in rows[header_row_index].find_all(['th', 'td']):
            if cell.has_attr('colspan'):
                if cell.has_attr('rowspan'):
                    for i in range(int(cell.get('colspan'))):
                        header_rowspan[indices[t + i]] = int(cell.has_attr('rowspan'))
                t += int(cell.get('colspan'))
            else:
                if cell.has_attr('rowspan'):
                    header_rowspan[indices[t]] = int(cell.has_attr('rowspan'))
                t += 1
        num_rowspans_in_header = max(header_rowspan.values())  # -= 1

    return table_headers, num_rows_in_header


def clean_html_table(html_table, url):
    rows = html_table.find_all('tr')
    rectangular_table = []
    table_headers, num_rows_in_header = process_table_headers(rows)
    rectangular_table.append(table_headers)

    num_of_headers = len(table_headers)
    row_span_text = {k: "" for k in range(num_of_headers)}
    row_span_number = {k: 1 for k in range(num_of_headers)}  #
    colspan_number = {k: 1 for k in range(num_of_headers)}
    for row in rows[num_rows_in_header:]:
        # assert(len(row.find_all(['th','td'])) == num_of_headers, f"Number of headers do not match the number of columns for this row {row}")

        # formulate new row with colspan info
        new_row = []

        # check for colspan in this row
        for indx, cell in enumerate(row.find_all(['th', 'td'])):
            if cell.has_attr('colspan'):
                spans_present = True
                colspan = int(cell.get('colspan'))
                new_row += [cell.text.strip()] * int(cell.get('colspan'))
            else:
                new_row.append(cell.text.strip())

        # forumlate new row with row span information
        cells_without_span = []
        columns_added = []
        indices_excluded = [k for k, v in row_span_number.items() if v > 1] + [k for k, v in colspan_number.items() if
                                                                               v > 1]
        for k, val in row_span_number.items():
            # if row span number is greater than 0, it means that value spans to this row
            if val > 1:
                new_row.insert(k, row_span_text[k].strip())
                row_span_number[k] -= 1
                # cell has both rowspan and columnspan from prev rows
                if colspan_number[k] > 1:
                    new_row[k:k] = [row_span_text[k].strip()] * (colspan_number[k] - 1)
                    columns_added.extend(list(range(k, colspan_number[k])))

                # change colspan to 1 if it becomes irrelevant to all further rows because of rowspan becoming 1
                if row_span_number[k] < 2:
                    colspan_number[k] = 1
                    cells_without_span.append(k)

            if len(new_row) == num_of_headers:
                rectangular_table.append(new_row)
            else:
                remove_sample = True
                print(
                        f'len(new_row) == num_of_headers for row: {new_row} and headers: {table_headers} ! Removing sample {url}')
                # raise Exception(f"Number of cells: {len(new_row)} is not equal to number of headers: {num_of_headers} for table {url}")

            # update row span info if the current row has the rowspan in the table data
            col_numbers_without_span = [k for k, v in row_span_number.items() if v < 2]
            assert ((len(col_numbers_without_span)-len(columns_added)-len(col_numbers_without_span)) == len(row.find_all(['td', 'th'])),
                    "row_span_number with 1s do not match columns without spans for this row!")
            #l3 = [x for x in col_numbers_without_span if x not in columns_added and x not in cells_without_span]
            l3 = [x for x in range(num_of_headers) if x not in indices_excluded]
            for td, idx in zip(row.find_all(['th', 'td']), l3):
                if td.get('rowspan'):
                    spans_present=True
                    row_span_text[idx] = td.text.strip()
                    row_span_number[idx] = int(td.get('rowspan'))
                    if td.get('colspan'):
                        colspan_number[idx] = int(td.get('colspan'))

        if None in rectangular_table:
            print(f'{url}  has None in table!!!!')
        for row in rectangular_table:
            if num_of_headers != len(row):
                print(f'{url} has non-rectangular table!!!')

    return rectangular_table

