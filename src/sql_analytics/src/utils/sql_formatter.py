from sqlparse import format as sql_format
from sqlparse import parse, format
from sqlparse.tokens import Keyword, DML
from sqlparse.sql import IdentifierList, Identifier, Comparison, Where, Parenthesis, Token, Function, Operation, Where, Case
from sql_metadata import Parser as meta_parser
from pathlib import Path
import re

def read_sql_file(path):
    """Reads SQL file and returns SQL query as string"""
    sql_list = []
    with open(path, 'r') as file:
        lines = file.readlines()
        sql_parts = []
        for line in lines:
            if line.startswith('--'):
                comment = line.strip()
            elif line.strip().endswith(';'):
                sql_parts.append(line.strip())
                sql_parts = ' '.join(sql_parts)
                sql_list.append({
                    'query': sql_parts,
                    'question': comment[2:],
                })
            else:
                if line.strip() != '':
                    sql_parts.append(line.strip())
    return sql_list

def format_sql(sql, reindent=False):
    """Formats SQL query to be more human-readable"""
    # reduce \n and space
    sql = re.sub(r'\s+', ' ', sql)  # reduce spaces
    sql = re.sub(r'\n', '', sql)  # reduce newlines
    sql = sql_format(sql, reindent_aligned=reindent, keyword_case='upper', use_space_around_operators=True)
    return sql

# need to test this function

def get_tokens(sql):
    """Returns a list of all identifiers in SQL query"""
    sql = format_sql(sql)
    parsed = parse(sql)[0]
    tokens = []
    
    for token in parsed.tokens:
        if token.ttype is Keyword:
            identifiers.append(token.value)
        if token.ttype is Identifier:
            # check if identifier is a subquery
            
            # check if identifier is a function
            
            identifiers.append(token.value)
    return identifiers

def count_join(sql):
    """Counts the number of joins in a SQL query"""
    joins = 0
    for t in sql.split(' '):
        if t.upper() == 'JOIN':
            joins += 1
    return joins

def count_subquery(sql):
    """Counts the number of subqueries in a SQL query"""
    subqueries = 0
    for t in sql.split(' '):
        if t.upper() == 'SELECT' or t.upper() == '(SELECT':
            subqueries += 1
    return subqueries - 1

def count_with(sql):
    """Counts the number of WITH statements in a SQL query"""
    withs = 0
    for t in sql.split(' '):
        if t.upper() == 'WITH':
            withs += 1
            break
    return withs

def count_case(sql):
    """Counts the number of CASE statements in a SQL query"""
    cases = 0
    for t in sql.split(' '):
        if t.upper() == 'CASE':
            cases += 1
    return cases

def do_contain_aggregation(sql):
    """Checks if SQL query contains aggregation"""
    return 'SUM(' in sql or 'COUNT(' in sql or 'AVG(' in sql or 'MIN(' in sql or 'MAX(' in sql

"""
test functions
"""
def test_identifiers():
    _sql = """SELECT
    CASE
        WHEN T.monat IN (12, 1, 2) THEN 'Winter'
        WHEN T.monat IN (3, 4, 5) THEN 'Frühling'
        WHEN T.monat IN (6, 7, 8) THEN 'Sommer'
        WHEN T.monat IN (9, 10, 11) THEN 'Herbst'
    END AS jahreszeit,
    SUM(T.pm10_ug_m3) / COUNT(T.pm10_ug_m3) AS avg_pm10_ug_m3,
    SUM(T.pm2_5_ug_m3) / COUNT(T.pm2_5_ug_m3) AS avg_pm2_5_ug_m3
FROM stadt_zurich_monatlich_luftqualitatsmessungen_seit_1983 AS T
JOIN spatial_unit AS S ON T.spatialunit_uid = S.spatialunit_uid
WHERE S.municipal = TRUE
    AND S.name = 'Zürich'
    AND T.jahr = 2018
    AND T.pm10_ug_m3 IS NOT NULL
    AND T.pm2_5_ug_m3 IS NOT NULL
GROUP BY jahreszeit;"""
    sql = """SELECT 100 * SUM(CASE WHEN T.total_gwh_2015 > T.total_gwh_2021 THEN 1 ELSE 0 END)/CAST(COUNT(*) AS FLOAT) AS prozent_abnahme
FROM (
    SELECT
        S.name,
        SUM(CASE WHEN T.jahr = 2015 THEN T.total_gwh ELSE 0 END) AS total_gwh_2015,
        SUM(CASE WHEN T.jahr = 2021 THEN T.total_gwh ELSE 0 END) AS total_gwh_2021
    FROM thurgau_erneuerbare_elektrizitatsproduktion_gemeinde AS T
    JOIN spatial_unit AS S ON T.spatialunit_uid = S.spatialunit_uid
    WHERE S.municipal = TRUE
        AND T.jahr IN (2015, 2021)
    GROUP BY S.name
    ) AS T;"""
    
    return get_tokens(format(sql, use_space_around_operators=True))

def normalize_spaces(match_obj):
    for i, _ in enumerate(match_obj.groups(), 1):
        if match_obj.group(i) is not None and match_obj.group(i+2) is not None:
            return match_obj.group(i) + ' ' + match_obj.group(i+1).strip() + ' ' + match_obj.group(i+2)

def normalize_as(match_obj):
    reserved_keywords = ['as', 'where', 'order', 'on'
                         'group', 'limit', 'join', 'having']
    # print(match_obj.groups())
    for i, grp in enumerate(match_obj.groups(), 1):
        if match_obj.group(i) is not None and match_obj.group(i+1) is not None and match_obj.group(i+2) is not None and match_obj.group(i+2).lower() not in reserved_keywords:
            #print(i, match_obj.group(i), match_obj.group(i+1))
            if match_obj.group(i+2):
                return match_obj.group(i) + ' ' + match_obj.group(i+1).rstrip() + ' AS ' + match_obj.group(i+2).strip()
        elif match_obj.group(i) is not None and match_obj.group(
                i+1) is not None and match_obj.group(i+2) is not None:
            return match_obj.group(i) + ' ' + match_obj.group(i+1).rstrip() + ' ' + match_obj.group(i+2).strip()

def quotate_boolean_values(match_obj):
    for i, _ in enumerate(match_obj.groups(), 1):
        if match_obj.group(i) is not None:
            # print(match_obj)
            return '\'' + match_obj.group(i).strip('\'') + '\''
    
def _add_spaces(query):
    operators_1 = ['\+', '\-', '/', '\*', '\=', '>', '<']
    operators_2 = ['>=', '<=', '!=', '<>']
    ops = operators_2 + operators_1
    re_patterns_list = [
        f"(\w+\.?\w*)(\s*{op}\s*)([\"\'\-\w]+\.?[\"\'\w]*)" for op in ops]
    regex = ('|').join(re_patterns_list)
    try:
        new_query = re.sub(regex, normalize_spaces, query)
    except Exception as e:
        print(f"_add_spaces: {query}")
        print(e)
        new_query = query
    return new_query

def _add_as(query):
    regex = re.compile(
        '(FROM)\s+([\w\_]+)\s+([\w\_]+)|(JOIN)\s+([\w\_]+)\s+([\w\_]+)', flags=re.IGNORECASE)
    try:
        new_query = re.sub(regex, normalize_as, query)
    except Exception as e:
        print(f"_add_as: {query}")
        print(e)
        new_query = query
    return new_query

def _add_quotes(query, keywords=['true', 'false', 'TRUE', 'FALSE']):

    _keywords = [f'\'{word}\'' for word in keywords]
    # keywords = _keywords + keywords
    if isinstance(query, str):
        for _k, k in zip(_keywords, keywords):
            query = query.replace(_k, k)
    regex = re.compile(r'\b(%s)\b' % '|'.join(keywords),
                       flags=re.IGNORECASE | re.MULTILINE)
    # print(regex)
    try:
        new_query = re.sub(regex, quotate_boolean_values, query)
    except Exception as e:
        print(f"_add_quotes: {query}")
        print(e)
        new_query = query

    return new_query

def query_cleaning(query):
    """todo:
    – SELECT * FROM A a JOIN B b ON A.a=B.b WHERE
    A.c = true
    + SELECT * FROM A AS a JOIN B AS b ON A.a = B.b
    WHERE A.c = ’true’ (use keyword "AS" explicitly, space
    before and after "=", and stringfy the boolean value
    "true"/"false")
    """
    # print(query)
    res = _add_quotes(_add_as(_add_spaces(query)))
    return res


def main():
    print(test_identifiers())

if __name__ == '__main__':
    main()


