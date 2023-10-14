import random
import argparse
import json
import jsonlines
import pandas as pd
from collections import defaultdict
from sqlalchemy import create_engine

from extract_wikitables import WikiTable
from utility import numeric_column_types

squall_path="data/squall.json"
squall = json.load(open(squall_path))
all_sql = [sample['sql'] for sample in squall]
number_sql_templates = []
no_number_sql_templates = []
for i, sample in enumerate(all_sql):
    contains_number = False
    for word in sample:
        if word[0] == 'Column':
            if 'number' in word[1] :
                number_sql_templates.append((i, sample))
                contains_number = True
                break
            if 'number' in word[0].lower():
                number_sql_templates.append((i, sample))
                contains_number = True
                break
    if not contains_number:
        no_number_sql_templates.append((i, sample))

print("number_sql_templates: ", len(number_sql_templates))
def is_numeric_type(word):
    for col_type in numeric_column_types:
        if col_type in word:
            return True
    return False

def join_sql(word_list):
    agg = ['count', 'sum', 'min', 'max', 'avg', 'abs', 'subtract']
    prev_word = None
    agg_started = False
    final_sql = []
    agg_phrase = ""
    for i, word in enumerate(word_list):
        if agg_started:
            if word == ")":
                agg_phrase += word
                agg_started = False
                final_sql.append(agg_phrase)
                agg_phrase = ""
            else:
                agg_phrase += word
        elif word in agg:
            agg_started = True
            agg_phrase += word
        else:
            final_sql.append(word)
    return final_sql

def get_database_numeric_columns(unique_tables, database_path="mariadb://root@localhost:5444/bengali_db"):
    engine = create_engine(database_path)
    connection = engine.connect()
    unique_tables_numeric_columns = defaultdict(list)
    for wikipage_name, tables in unique_tables.items():
        for table_name in tables:
            table = pd.read_sql(f'SELECT * FROM `{table_name}`', connection)
            numeric_columns = []
            for column in table.columns[1:]:
                is_column_numeric = True
                for val in table[column]:
                    if not val.isnumeric():
                        is_column_numeric = False
                        break
                if is_column_numeric:
                    numeric_columns.append(column)
            unique_tables_numeric_columns[wikipage_name].append(numeric_columns)
    connection.close()
    return unique_tables_numeric_columns

def create_codemixed_sql_instances(unique_tables,
                                   output_path="data/bengali_sql/non_numeric_code_mixed.jsonl",
                                   database_path="mariadb://root@localhost:5444/bengali_db",
                                   split="non_numeric"):
    correction = {'count ( ': 'count(', "sum ( ": 'sum(', 'min ( ': 'min(', 'max ( ': 'max('}
    sql_instances = []
    unique_tables_numeric_columns = get_database_numeric_columns(unique_tables=unique_tables, database_path=database_path)
    tables_to_sql_instances = defaultdict(list)
    connection=create_engine(database_path)
    if split == "non_numeric":
        templates = no_number_sql_templates
    elif split == "numeric":
        templates = number_sql_templates
    else:
        templates = no_number_sql_templates
    with jsonlines.open(output_path, "w", flush=True) as f:
        for wikipage_name, tables in unique_tables.items():
            numeric_columns_of_tables = unique_tables_numeric_columns[wikipage_name]
            print(numeric_columns_of_tables)
            for table_name, numeric_columns in zip(tables, numeric_columns_of_tables):

                try:
                    table = pd.read_sql(f'SELECT * FROM `{table_name}`', connection)
                    print(f"Processing table {table_name}")
                    df = table
                    for i, element in enumerate(templates):
                        index, sql = element
                        templ = " ".join([word[1] for word in sql])
                        if " id " in templ:
                            continue
                        sql_instance = []
                        column_variables_to_instances = defaultdict(str)
                        last_seen_column = None
                        invalid_sql = False
                        for word in sql:
                            #print(word)
                            if word[1] == 'w':  # table name
                                sql_instance.append(f"`{table_name}`")
                            elif word[0] == 'Column':  # instantiate non-numeric column-template with column name
                                if word[1] not in column_variables_to_instances.keys():
                                    try:
                                        if is_numeric_type(word[1]):  # column is numeric type
                                            numeric_column = random.choice(numeric_columns)
                                            column_variables_to_instances[word[1]] = numeric_column
                                        else:  # column is non-numeric
                                            column_variables_to_instances[word[1]] = random.choice(df.columns[1:])
                                    except Exception as e1:
                                        invalid_sql = True
                                        print(e1)
                                        break
                                        #continue
                                sql_instance.append(f'`{column_variables_to_instances[word[1]]}`')
                                last_seen_column = column_variables_to_instances[word[1]]
                            elif 'Literal' in word[0]:
                                try:
                                    literals = table[last_seen_column].to_list()
                                    literal = random.choice(table[last_seen_column].to_list())
                                    sql_instance.append(f'"{literal}"')
                                except:
                                    invalid_sql = True
                                    break
                                    #continue
                            else:
                                sql_instance.append(word[1])
                        if invalid_sql:
                            continue
                        corrected_sql = join_sql(sql_instance)
                        sql_instance_str = " ".join(corrected_sql)
                        if sql_instance_str.strip() in tables_to_sql_instances[table_name]:  # do not add duplicate query
                           continue
                        try:
                            print(sql_instance_str)
                            answer = pd.read_sql(sql_instance_str, connection)
                        except Exception as e:
                            print(e)
                            continue
                        if not answer.empty:
                            f.write({"sql": sql_instance_str, "answer": answer.to_json(orient='split'), "input_table_name": table_name, "input_tables": table.to_json(orient='split')})
                except Exception as e2:
                    print(e2)
                    continue
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
    all_unique_tables, unique_tables = wikitables.get_unique_tables(sql_database_path="mariadb://root@localhost:5444/bengali_db")
    create_codemixed_sql_instances(unique_tables=unique_tables,
                                   output_path="data/bengali_sql/numeric_code_mixed.jsonl",
                                   database_path="mariadb://root@localhost:5444/bengali_db",
                                   split="numeric")


if __name__ == "__main__":
    main():w

