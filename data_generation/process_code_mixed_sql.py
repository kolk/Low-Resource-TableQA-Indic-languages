from deep_translator import GoogleTranslator
import regex
import re
import pandas as pd
import time
from collections import defaultdict
import argparse
import jsonlines
from data.extract_wikitables import WikiTable

class CodeMixedSQL:
    def __init__(self, language, unique_tables):
        self.regex_bengali = r'\P{L}*\p{Bengali}+(?:\P{L}+\p{Bengali}+)*\P{L}*'
        self.regex_oriya = r'\P{L}*\p{Oriya}+(?:\P{L}+\p{Oriya}+)*\P{L}*'
        self.regex_assamese = r'\P{L}*\p{Bengali}+(?:\P{L}+\p{Bengali}+)*\P{L}*'
        self.regex_devanagari = r'\P{L}*\p{Devanagari}+(?:\P{L}+\p{Devanagari}+)*\P{L}*'
        self.language = language
        self.unique_tables = unique_tables
        self.translator = self.get_translator()
        self.bengali_sql_keyword = {"SELECT": "নির্বাচন করুন", "FROM": "থেকে", "WHERE": "যেখানে", "GROUP BY": "দল করা",
                           "ORDER BY": "সাজান হোক", "ASC": "ঊর্ধ্বগামী", "DESC": "অবরোহী", "avg": "গড়", "min": "সর্বনিম্ন",
                           "max": "সর্বোচ্চ", "sum": "যোগফল", "count": "গণনা",  "AVG": "গড়", "MIN": "সর্বনিম্ন",
                           "MAX": "সর্বোচ্চ", "SUM": "যোগফল", "COUNT": "গণনা", "AS": "হিসাবে", "JOIN": "সংযুক্ত করা",
                           "ON": "উপর", "IN": "মধ্যে", "NOT": "নয়", "BETWEEN": "মধ্যে", "AND": "এবং", "OR": "অথবা",
                           "HAVING": "যার আছে", "UNION": "সংযোগ", "INTERSECT": "ছেদ", "EXCEPT": "বাদে", "LIMIT": "সীমা",
                           "LIKE": "অনুরূপ", "DISTINCT": "অনন্য"}
        self.assamese_sql_keyword = {"SELECT": "নচয়ন কৰা", "FROM": "পৰা", "WHERE": "ক’ত", "GROUP BY": "টিম আপ কৰক",
                                "ORDER BY": "সাজান হোক", "ASC": "আৰোহী", "DESC": "অৱতৰণ কৰা", "avg": "গড়",
                                "min": "নূন্যতম",
                                "max": "সৰ্বোচ্চ", "sum": "মুঠ", "count": "হিচাপ কৰা", "AVG": "গড়", "MIN": "নূন্যতম",
                                "MAX": "সর্বোচ্চ", "SUM": "মুঠ", "COUNT": "হিচাপ কৰা", "AS": "যেনেকৈ",
                                "JOIN": "যোগদান কৰক",
                                "ON": "ওপৰত", "IN": "ভিতৰত", "NOT": "নহয়", "BETWEEN": "মাজত", "AND": "আৰু",
                                "OR": "অথবা",
                                "HAVING": "যাৰ আছে", "UNION": "সংযোগ", "INTERSECT": "ছেদ", "EXCEPT": "ইয়াৰ বাহিৰে",
                                "LIMIT": "সীমাবদ্ধ",
                                "LIKE": "ৰ সৈতে একেধৰণৰ", "DISTINCT": "অনন্য"}
        self.hindi_sql_keyword = {"SELECT": "चुनना", "FROM": "से", "WHERE": "जहां", "GROUP BY": "दल बनाइये",
                           "ORDER BY": "व्यवस्थित करना", "ASC": "आरोही", "DESC": "अवरोही", "avg": "औसत", "min": "न्यूनतम",
                           "max": "अधिकतम", "sum": "जोड़", "count": "गणना",  "AVG": "औसत", "MIN": "न्यूनतम",
                           "MAX": "अधिकतम", "SUM": "जोड़", "COUNT": "गणना", "AS": "औसत", "JOIN": "संलग्न करना",
                           "ON": "पर", "IN": "में", "NOT": "नहीं", "BETWEEN": "मध्य में", "AND": "और", "OR": "या",
                           "HAVING": "जिसके पास", "UNION": "संयोजन", "INTERSECT": "प्रतिच्छेद", "EXCEPT": "सिवाय", "LIMIT": "सीमा",
                           "LIKE": "अनुरूप", "DISTINCT": "अनन्य"}
        self.oriya_sql_keyword = {"SELECT": "ଚୟନ କରନ୍ତୁ", "FROM": "ଠାରୁ", "WHERE": "କେଉଁଠାରେ", "GROUP BY": "ଦଳ କରିବା",
                           "ORDER BY": "ବ୍ୟବସ୍ଥିତ କରିବା", "ASC": "ଉର୍ଦ୍ଧ୍ୱଗାମୀ", "DESC": "ଅବରୋହୀ", "avg": "ହାରାହାରି", "min": "ସର୍ବନିମ୍ନ",
                           "max": "ସର୍ବାଧିକ", "sum": "ଯୋଗଫଳ", "count": "ଗଣନା",  "AVG": "ହାରାହାରି", "MIN": "ସର୍ବନିମ୍ନ",
                           "MAX": "ସର୍ବାଧିକ", "SUM": "ରାଶି", "COUNT": "ଗଣନା", "AS": "ହିସାବରେ", "JOIN": "ସଂଲଗ୍ନ କରିବା",
                           "ON": "ଉପରେ", "IN": "ରେ", "NOT": "ନୁହେଁ", "BETWEEN": "ମଝିରେ", "AND": "ଏବଂ", "OR": "କିମ୍ବା",
                           "HAVING": "ଯାହା ପାଖରେ", "UNION": "ସଂଯୋଗ", "INTERSECT": "ପ୍ରତିଚ୍ଛେଦ", "EXCEPT": "ଏହା ବ୍ୟତୀତ", "LIMIT": "ସୀମା",
                           "LIKE": "ଅନୁରୂପ", "DISTINCT": "ଅନନ୍ୟ"}
        self.marathi_sql_keyword = {"SELECT": "ଚୟନ କରନ୍ତୁ", "FROM": "ଠାରୁ", "WHERE": "କେଉଁଠାରେ", "GROUP BY": "ଦଳ କରିବା",
                           "ORDER BY": "ବ୍ୟବସ୍ଥିତ କରିବା", "ASC": "ଉର୍ଦ୍ଧ୍ୱଗାମୀ", "DESC": "ଅବରୋହୀ", "avg": "ହାରାହାରି", "min": "ସର୍ବନିମ୍ନ",
                           "max": "ସର୍ବାଧିକ", "sum": "ଯୋଗଫଳ", "count": "ଗଣନା",  "AVG": "ହାରାହାରି", "MIN": "ସର୍ବନିମ୍ନ",
                           "MAX": "ସର୍ବାଧିକ", "SUM": "ରାଶି", "COUNT": "ଗଣନା", "AS": "ହିସାବରେ", "JOIN": "ସଂଲଗ୍ନ କରିବା",
                           "ON": "ଉପରେ", "IN": "ରେ", "NOT": "ନୁହେଁ", "BETWEEN": "ମଝିରେ", "AND": "ଏବଂ", "OR": "କିମ୍ବା",
                           "HAVING": "ଯାହା ପାଖରେ", "UNION": "ସଂଯୋଗ", "INTERSECT": "ପ୍ରତିଚ୍ଛେଦ", "EXCEPT": "ଏହା ବ୍ୟତୀତ", "LIMIT": "ସୀମା",
                           "LIKE": "ଅନୁରୂପ", "DISTINCT": "ଅନନ୍ୟ"}
        self.malayalam_sql_keyword = {"SELECT": "ଚୟନ କରନ୍ତୁ", "FROM": "ଠାରୁ", "WHERE": "କେଉଁଠାରେ", "GROUP BY": "ଦଳ କରିବା",
                                    "ORDER BY": "ବ୍ୟବସ୍ଥିତ କରିବା", "ASC": "ଉର୍ଦ୍ଧ୍ୱଗାମୀ", "DESC": "ଅବରୋହୀ",
                                    "avg": "ହାରାହାରି", "min": "ସର୍ବନିମ୍ନ",
                                    "max": "ସର୍ବାଧିକ", "sum": "ଯୋଗଫଳ", "count": "ଗଣନା", "AVG": "ହାରାହାରି",
                                    "MIN": "ସର୍ବନିମ୍ନ",
                                    "MAX": "ସର୍ବାଧିକ", "SUM": "ରାଶି", "COUNT": "ଗଣନା", "AS": "ହିସାବରେ",
                                    "JOIN": "ସଂଲଗ୍ନ କରିବା",
                                    "ON": "ଉପରେ", "IN": "ରେ", "NOT": "ନୁହେଁ", "BETWEEN": "ମଝିରେ", "AND": "ଏବଂ",
                                    "OR": "କିମ୍ବା",
                                    "HAVING": "ଯାହା ପାଖରେ", "UNION": "ସଂଯୋଗ", "INTERSECT": "ପ୍ରତିଚ୍ଛେଦ",
                                    "EXCEPT": "ଏହା ବ୍ୟତୀତ", "LIMIT": "ସୀମା",
                                    "LIKE": "ଅନୁରୂପ", "DISTINCT": "ଅନନ୍ୟ"}

        self.bengali_sql_keywords_sorted = self.sort_sql_keywords(self.bengali_sql_keyword)
        self.assamese_sql_keywords_sorted = self.sort_sql_keywords(self.assamese_sql_keyword)
        self.oriya_sql_keywords_sorted = self.sort_sql_keywords(self.oriya_sql_keyword)
        self.hindi_sql_keywords_sorted = self.sort_sql_keywords(self.hindi_sql_keyword)

    def get_language_regex(self):
        if self.language == "bengali":
            return self.regex_bengali
        elif self.language == "assamese":
            return self.regex_assamese
        elif self.language == "hindi":
            return self.regex_devanagari
        elif self.language == "oriya" or self.language == "odia":
            return self.regex_oriya

    def get_translator(self):
        if self.language == "bengali":
            return GoogleTranslator(source="en", target="bn")
        elif self.language == "assamese":
            return GoogleTranslator(source="en", target="as")
        elif self.language == "hindi":
            return GoogleTranslator(source="en", target="hi")
        elif self.language == "oriya" or self.language == "odia":
            return GoogleTranslator(source="en", target="or")

    def sort_sql_keywords(self, sql_keyword):
        sql_keywords_lower_case = {k.lower(): v for k, v in sql_keyword.items()}
        sql_keyword.update(sql_keywords_lower_case)
        return sorted(list(sql_keyword.keys()), key=lambda x: len(x), reverse=True)

    def get_sql_keywords_dictionary(self):
        if self.language == "bengali":
            return self.bengali_sql_keywords_sorted, self.bengali_sql_keyword
        elif self.language == "assamese":
            return self.assamese_sql_keywords_sorted, self.assamese_sql_keyword
        elif self.language == "hindi":
            return self.hindi_sql_keywords_sorted, self.hindi_sql_keyword
        elif self.language == "odia" or self.language == "oriya":
            return self.oriya_sql_keywords_sorted, self.oriya_sql_keyword

    def get_digit_translation_dictionary(self):
        bn2enDigits = {'০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4', '৫': '5', '৬': '6', '৭': '7', '৮': '8',
                       '৯': '9',
                       '.': '.'}
        en2bnDigits = {v: k for k, v in bn2enDigits.items()}

        as2enDigits = {'০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4', '৫': '5', '৬': '6', '৭': '7', '৮': '8',
                       '৯': '9',
                       '.': '.'}
        en2asDigits = {v: k for k, v in as2enDigits.items()}

        hi2enDigits = {'०': '0', '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8',
                       '९': '9', '.': '.'}
        en2hiDigits = {v: k for k, v in hi2enDigits.items()}

        or2enDigits = {'୦': '0', '୧': '1', '୨': '2', '୩': '3', '୪': '4', '୫': '5', '୬': '6', '୭': '7', '୮': '8',
                       '୯': '9', '.': '.'}
        en2orDigits = {v: k for k, v in hi2enDigits.items()}

        if self.language == "bengali":
            return bn2enDigits, en2bnDigits
        elif self.language == "assamese":
            return as2enDigits, en2asDigits
        elif self.language == "hindi":
            return hi2enDigits, en2hiDigits
        elif self.language == "oriya" or self.language == "odia":
            return or2enDigits, en2orDigits
        else:
            en2enDigits = {k:k for k, v in en2bnDigits.items()}
            return en2enDigits, en2enDigits

    def translate_english_number(self, english_number):
        indic2enDigits, en2IndicDigits = self.get_digit_translation_dictionary()
        indic2enDigits.update({'.': '.', '-': '-', '?': '?', '(': '(', ')': ')'})
        en2IndicDigits.update({'.': '.', '-': '-', '?': '?', '(': '(', ')': ')'})
        indic_number = []
        for english_digit in english_number:
            if english_digit in en2IndicDigits.keys():
                indic_number.append(en2IndicDigits[english_digit.strip()])
            else:
                indic_number.append(english_digit)
        return "".join(indic_number)

    def translate_digits_in_question(self, question):
        new_query = []
        for val in question.split():
            # check for numeric
            is_digit = r"""^[(-.:"'`]*\d+[-:,.]*\d+[)-:%.]*[?'"`]*"""
            if regex.fullmatch(is_digit, val):  # numbers like 3342-2343, 34234, (3432)?
                new_query.append(self.translate_english_number(val))
            else:
                new_query.append(val)
        return " ".join(new_query)

    def get_sql_without_table_name(self, query, table_name):
        if self.language == "bengali" or self.language == "bangla":
            return query.replace(f'থেকে `{table_name}`', "")
        elif self.language == "assamese":
            return query.replace(f'পৰা `{table_name}`', "")
        elif self.language == "hindi":
            return query.replace(f'से `{table_name}`', "")
        elif self.language == "oriya" or self.language == "odia":
            return query.replace(f'ଠାରୁ `{table_name}`', "")
        else:
            return query

    def translate_table_column(self, column):
        sql_keywords_sorted, sql_keyword = self.get_sql_keywords_dictionary()
        new_column = column
        for k in sql_keywords_sorted:
            if k in column:
                new_column = new_column.replace(k, sql_keyword[k])
        return new_column

    def translate_answer_table(self, answer):
        RE_D = re.compile('\d')
        answer = pd.read_json(answer, orient='split')
        if not answer.empty:
            # translate answer table digits to indic language
            answer = answer.applymap(
                lambda x: self.translate_english_number(str(x))
                    if isinstance(x, int)
                       or isinstance(x, float)
                       or RE_D.match(str(x))
                    else x)
            return answer.rename(columns={col:self.translate_table_column(col) for col in list(answer.columns)})
        else:
            return None

    def convert_code_mixed_sql_to_indic_sql(self,
                                            input_path,
                                            output_path,
                                            input_file_start_index=0,
                                            input_file_end_index=None):
        sql_keywords_sorted, sql_keyword = self.get_sql_keywords_dictionary()
        language_regex = self.get_language_regex()
        table_names = set([table_name for prefix, tables in self.unique_tables.items() for table_name in tables])
        indic_sql = []
        translated_dictionary = defaultdict(str)
        with jsonlines.open(input_path) as f, \
            jsonlines.open(output_path, "a", flush=True) as f_w:
            for i, sample in enumerate(f):
                if i <= input_file_start_index:
                    continue
                query = sample['sql']

                # translate query keywords
                for keyword in sql_keywords_sorted:
                    query = query.replace(keyword, sql_keyword[keyword])

                # translate all digits
                query = self.translate_digits_in_question(query)

                #sql without table name
                sql_without_tables = query
                for table_name in table_names:
                    if table_name in query:
                        sql_without_tables = self.get_sql_without_table_name(query, table_name)
                        break

                # translate rest of english words
                full_indic = []
                for word in query.split():
                    table_name_match = regex.fullmatch(".*_[0-9]*[\"'`]*", word)
                    if table_name_match: # remove _122324 from table name
                        word = " ".join(word.split('_')[:-1])

                    if regex.fullmatch(language_regex, word):
                        full_indic.append(word)
                    else:
                        try:
                            if word not in translated_dictionary.keys():
                                translated_word = self.translator.translate(word)
                                time.sleep(0.001)
                            else:
                                translated_word = translated_dictionary[word]
                            if translated_word:
                                full_indic.append(translated_word)
                            else:
                                full_indic.append(word)
                        except Exception as e:
                            print(e)
                            full_indic.append(word)

                query = " ".join(full_indic)

                # translate query without table name to english
                full_indic_without_table_name = []
                for word in sql_without_tables.split():
                    if regex.fullmatch(language_regex, word):
                        full_indic_without_table_name.append(word)
                    else:
                        try:
                            if word not in translated_dictionary.keys():
                                translated_word = self.translator.translate(word)
                                time.sleep(0.001)
                            else:
                                translated_word = translated_dictionary[word]
                            if translated_word:
                                full_indic_without_table_name.append(translated_word)
                            else:
                                full_indic_without_table_name.append(word)
                        except Exception as e:
                            print(e)
                            full_indic_without_table_name.append(word)
                query_full_indic_without_table_name = " ".join(full_indic_without_table_name)

                # translate answer table
                translated_answer = self.translate_answer_table(sample['answer'])
                sample[f"{self.language}_sql"] = query
                sample['sql_without_table_name'] = query_full_indic_without_table_name
                sample["index"] = i
                sample["translated_answer"] = translated_answer.to_json(orient="split", force_ascii=False)
                f_w.write(sample)

    def flatten_dataset(self, line):
        database_path = 'data/database'
        # con = sqlite3.connect(os.path.join(database_path, line["db_name"], line["db_name"] + '.sqlite'))
        # encoding = "latin1"
        # con.text_factory = lambda x: str(x, encoding)
        # tables = [pd.read_sql_query(f'SELECT * FROM {table_name}', con) for table_name in line["tables"]]
        return self.preprocess_sample(
            {"query": line["bengali_natural_question"], "tables": line["input_tables"], "answer": line["translated_answer"]})



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/bengali_sql/non_numeric_code_mixed.jsonl", help="path to input jsonl file")
    parser.add_argument("--output_file", type=str, default="data/bengali_sql/non_numeric_full_indic.jsonl", help="path to output jsonl file")
    parser.add_argument("--table_language", default="bn", type=str, help="Language of Wikipedia from which to extract tables")
    parser.add_argument("--sql_language", default="bengali", type=str, help="Language of code-mixed SQL")


    args = parser.parse_args()
    wikitables = WikiTable()
    all_unique_tables, unique_tables = wikitables.get_unique_tables(
        sql_database_path="mariadb://root@localhost:5444/bengali_db")
    codemixed_sql_processor = CodeMixedSQL(language=args.sql_language, unique_tables=unique_tables)
    codemixed_sql_processor.convert_code_mixed_sql_to_indic_sql(input_path=args.input_file,
                                                                output_path=args.output_file,
                                                                input_file_start_index=0,
                                                               )


if __name__ == "__main__":
    main()



