import json
from collections import defaultdict
from datasets import load_dataset
from bs4 import BeautifulSoup
import wikipediaapi
import requests
import csv
from data.parse_wikitable import clean_html_table
import jsonlines
from sqlalchemy import create_engine
import pandas as pd
import os
import sqlite3
import pandasql as ps
import argparse
from collections import defaultdict
import mysql

def clean_table(html_table):
    html_table
table_class = ['wikitable sortable', 'wikitable collapsible', 'wikitable collapsed']
hi_urls = defaultdict(list)

class WikiTable:
    def __init__(self,
                path="data/bengali_tables.jsonl",
                language_code="bn",
                dump_date="20220120",
                max_table_cells=500
                ):
        self.path = path
        self.language_code = language_code
        self.max_table_size = max_table_cells
        self.wiki_dump_date = dump_date

    def extract_wikipages(self):
        wikipedia = load_dataset("wikipedia", language=self.language_code, date=self.dump_date, beam_runner='DirectRunner')
        self.titles = []
        for page in wikipedia['train']:
            self.titles.append(page['title'])
        del (wikipedia)

    def extract_tables(self):
        wiki_html = wikipediaapi.Wikipedia(
            language=self.language_code,
        )

        with jsonlines.open(self.path, "w", flush=True) as f:
            for i, title in enumerate(self.titles):
                try:
                    url = wiki_html.page(title).fullurl
                    page = requests.get(url).text
                    soup = BeautifulSoup(page, 'html.parser')
                    for tab_class in table_class:
                        table = soup.find('table', class_=tab_class)
                        if table:
                            rectangular_table = clean_html_table(table, url)
                            if len(rectangular_table[1:]) * len(rectangular_table[0]) <= self.max_table_size:
                                f.write({"title": title, "url": url, "table": rectangular_table})
                except:
                    continue


    def convert_tables_to_pandas(self, database_path="mariadb://root@localhost:5444/bengali_db"):
        unique_titles = set([])
        tables = []
        titles_to_tables = defaultdict(list)
        engine = create_engine(database_path)
        connection = engine.connect()
        table_index = 0
        with jsonlines.open(self.path) as f:
            for i, line in enumerate(f):

                # table name indexing for multiple tables under same title
                if line["title"] in unique_titles:
                    table_index += 1
                else:
                    table_index = 0
                    unique_titles.add(line["title"])

                header = line["table"][0]
                table = line["table"][1:]
                new_table = []

                # only keep rectangular tables
                ignore_sample = False
                for row in table:
                    if len(row) != len(header):
                        ignore_sample = True
                        if len(row) > len(header):
                            num_of_extra_cells = len(row) - len(header)
                            new_row = []
                            for j, elem in enumerate(row[::-1]):
                                if j <= (num_of_extra_cells - 1) and elem == "":
                                    continue
                                new_row.append(elem)
                            print(new_row)
                            new_table.append(new_row)
                    else:
                        new_table.append(row)

                # table is rectangular
                if not ignore_sample:
                    tables.append(pd.DataFrame(table, columns=header))
                    df = pd.DataFrame(table, columns=header)
                    table_name = f"{line['title']}_{table_index}"
                    titles_to_tables[table_name].append(df)
                    try:
                        df.to_sql(con=connection, name=table_name, if_exists='replace')
                    except:
                        continue
                else:
                    continue
        connection.close()

    def get_unique_tables(self, sql_database_path="mariadb://root:@localhost:5444/bengali_db"):
        def is_unique(table_name, all_tables):
            for table_nm in all_tables:
                if pd.read_sql(f'SELECT * FROM `{table_name}`', connection).equals(
                        pd.read_sql(f'SELECT * FROM `{table_nm}`', connection)):
                    return False
            return True

        engine = create_engine(sql_database_path)
        connection = engine.connect()
        all_bengali_table_names = pd.read_sql('SHOW tables', connection)
        unique_bengali_tables = defaultdict(list)
        all_unique_bengali_tables = []
        for i in range(len(all_bengali_table_names)):
            table_name = all_bengali_table_names.iloc[i][0]
            table_name_prefix = table_name[:table_name.rfind('_')]
            if is_unique(table_name, unique_bengali_tables[table_name_prefix]):
                all_unique_bengali_tables.append(table_name)
                unique_bengali_tables[table_name_prefix].append(table_name)
        return all_unique_bengali_tables, unique_bengali_tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table_language", default="as", type=str, help="Language of Wikipedia from which to extract tables")
    parser.add_argument("--data_save_path", default="data/bengali_tables.jsonl", type=str, help="jsonlines file in which to save the tables")
    parser.add_argument("--wiki_dump_date", default="date", type=str, help="date of Wikipedia dump")
    parser.add_argument("--max_table_cells", default=500, type=int, help="maximum number of cells of extracted table to be kept")
    args = parser.parse_args()
    wikitables = WikiTable(path=args.data_save_path,
                           language_code=args.table_language,
                           dump_date=args.wiki_dump_date,
                           max_table_cells=args.max_table_cells)
    #wikitables.extract_tables()
    #wikitables.convert_tables_to_pandas(database_path=f"mariadb://root@localhost:5444/bengali_db")
    all_unique_tables, unique_tables = wikitables.get_unique_tables(sql_database_path="mariadb://root:@localhost:5444/bengali_db")
    print(all_unique_tables)
    print(unique_tables)

if __name__ == "__main__":
    main()
