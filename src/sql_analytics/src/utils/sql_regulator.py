from dotenv import load_dotenv
from sql_metadata import Parser
from sqlparse import parse
from sqlparse.sql import (
    IdentifierList,
    Identifier,
    Parenthesis,
    Token,
    Operation,
    Where,
    Comparison,
    Function,
)
from sqlparse.tokens import Token as T
from utils.sql_database import SQLDatabase

from sentence_transformers import SentenceTransformer, util


# for multilingual support using "paraphrase-multilingual-MiniLM-L12-v2"
# for English support using "paraphrase-MiniLM-L6-v2"
class SQLRegulator(object):
    def __init__(
        self,
        sql_database: SQLDatabase,
        incl_tables=None,
        model="paraphrase-MiniLM-L6-v2",
        sql=None,
    ):
        load_dotenv()
        self.db = sql_database
        self.sql = sql
        self.parser = None
        self.sqlparser = None
        self.model = SentenceTransformer(model)
        self.table_info = self.db.get_table_info_dict(incl_tables)
        self.all_columns = self.get_all_columns(self.table_info)
        self.unmatched_columns = None
        if self.sql is not None:
            self.unmatched_columns = self._unmatched_columns()
            self.sqlparser = parse(self.sql)[0]

    def normalize_type(self, col_type):
        if "(" in col_type:
            col_type = col_type.split("(")[0]
        if col_type in ["VARCHAR", "CHAR", "TEXT"]:
            return "TEXT"
        if col_type in [
            "TINYINT",
            "BIGINT",
            "SMALLINT",
            "INTEGER",
            "REAL",
            "DOUBLE",
            "FLOAT",
            "DECIMAL",
            "NUMERIC",
            "TIMESTAMP",
        ]:
            return "NUMERIC"
        return col_type

    def get_all_columns(self, db_info_dict: dict):
        columns = []
        for table, v in db_info_dict.items():
            columns.extend(
                [[f"{table}.{col[0]}", self.normalize_type(col[1])] for col in v["COL"]]
            )
        return columns

    def _unmatched_columns(self):
        if self.sql is None:
            raise ValueError(
                "SQL is not set!\nPlease use update_sql() to set SQL first!"
            )
        found_columns = self.found_columns(self.sql)
        # print(found_columns)
        if "*" in found_columns:
            found_columns.remove("*")
        res = []
        all_columns_name = [c[0] for c in self.all_columns]
        for col in found_columns:
            if col not in all_columns_name:
                res.append(col)
        return res

    def update_sql(self, sql: str):
        self.sql = sql
        self.parser = Parser(self.sql)
        self.sqlparser = parse(self.sql)[0]
        self.unmatched_columns = self._unmatched_columns()
        return self

    def found_columns(self, sql: str):
        if self.parser:
            return self.parser.columns
        raise ValueError(
            "Parser is not set!\nPlease use update_sql() to set SQL first!"
        )

    """
    # @TODO: find the columns in where or having clause
    # So far it does not support math operations or functions
    def columns_in_condition(self):
        cols_dict = self.parser.columns_dict
        having_cols = cols_dict.get("having", [])
        where_cols = cols_dict.get("where", [])
        condition_cols = list(set(having_cols).union(set(where_cols)))

        alias_table_dict = self.parser.tables_aliases
        table_alias_dict = {v: k for k, v in alias_table_dict.items()}
        aliased_condition_cols = [
            ".".join(
                [
                    table_alias_dict.get(col.split(".")[0], col.split(".")[0]),
                    col.split(".")[1],
                ]
            )
            for col in condition_cols
        ]
        sql_tokens = [
            self.parser.tokens[i].value for i in range(len(self.parser.tokens))
        ]
        # print(sql_tokens, condition_cols, aliased_condition_cols)
        all_indices = [
            i for i, x in enumerate(sql_tokens) if x in aliased_condition_cols
        ]
        res = []
        for i in all_indices:
            if sql_tokens[i + 1] and sql_tokens[i + 1] in [
                "=",
                "!=",
                ">",
                "<",
                ">=",
                "<=",
            ]:
                res.append(sql_tokens[i : i + 3])
            elif sql_tokens[i + 1] and sql_tokens[i + 1].lower() in ["between"]:
                res.append(sql_tokens[i : i + 4])
            elif sql_tokens[i + 1] and sql_tokens[i + 1].lower() in ["like", "ilike"]:
                res.append(sql_tokens[i : i + 3])
            elif sql_tokens[i + 1] and sql_tokens[i + 1].lower() in ["not"]:
                if sql_tokens[i + 2] and sql_tokens[i + 2].lower() in ["between"]:
                    res.append(sql_tokens[i : i + 5])
                elif sql_tokens[i + 2] and sql_tokens[i + 2].lower() in [
                    "like",
                    "ilike",
                ]:
                    res.append(sql_tokens[i : i + 4])
                else:
                    raise ValueError(f"Unexpected token: {sql_tokens[i + 2]}")
        return res
    """

    # TODO

    def predicate_column_type(self):
        if self.sqlparser is None:
            raise ValueError(
                "SQL parser is not set!\nPlease use update_sql() to set SQL first!"
            )
        try:
            preparsed = parse_comparison(self.sqlparser)
            col_types = _get_all_columns(preparsed)
            col_types = list(filter(lambda x: x[1] != "unknown", col_types))
        except Exception as e:
            print(f"errors found by parsing column types\n{e.__traceback__}")
            return []

    def top_n_by_embedding_similarity(self, n=1, to_string: bool = False):
        """
        Return the top n most similar columns for each unmatched column by the embedding similarity.
        :param n: top n most similar columns
        the output is a [[[unmatched_columns[0], most_similar_columns[0], type_of_most_similar_columns, score], ...
                        [unmatched_columns[0], most_similar_columns[n], type_of_most_similar_columns, score]],
                        [[unmatched_columns[0], most_similar_columns[0], type_of_most_similar_columns, score], []]]
        :param to_string: if True, return a formatted string
        """
        if self.unmatched_columns is None:
            raise ValueError(
                "unmatched_columns is not set!\nPlease use update_sql() to set SQL first!"
            )
        all_colums_name = [c[0] for c in self.all_columns]
        mixed_emb = self.model.encode(
            self.unmatched_columns + all_colums_name, convert_to_tensor=True
        )
        unm_len = len(self.unmatched_columns)
        cos_scores = util.cos_sim(mixed_emb[:unm_len], mixed_emb[unm_len:])
        # print(cos_scores)
        max_indices = [(cos_scores * -1).argsort()[i][:n] for i in range(unm_len)]
        most_similar_columns = [
            [
                [
                    self.unmatched_columns[j],
                    *self.all_columns[i],
                    cos_scores[j].tolist()[i],
                ]
                if i < len(self.all_columns)
                else None
                for i in max_indices[j]
            ]
            for j in range(unm_len)
        ]
        if to_string:
            s = "**Top N most similar columns**\n"
            for a in most_similar_columns:
                for i, b in enumerate(a, start=1):
                    if b is not None:
                        s += f"For the unmatched columns [{b[0]}], found the {ordinal(i)} most similar columns [{b[1]}] in type of [{b[2]}] with similarity: ({b[3]})\n"
            return s
        return most_similar_columns


def parse_parenthesis(parenthesis_item):
    assert parenthesis_item[0].value == "(" and parenthesis_item[-1].value == ")"
    assert isinstance(parenthesis_item[1], Token)
    res = []
    for t in parenthesis_item:
        if isinstance(t, Identifier):
            res.append(t)
        elif isinstance(t, IdentifierList):
            res.append(parse_identifier_list(t))
        elif isinstance(t, Parenthesis):
            res.append(parse_parenthesis(t))
        else:
            res.append(parse_components(t)) if parse_components(t) else res
    return res


def parse_identifier_list(identifier_list):
    res = []
    assert isinstance(identifier_list, IdentifierList)
    for t in identifier_list:
        if isinstance(t, Identifier):
            res.append(t)
        elif isinstance(t, Operation):
            res.append(parse_operation(t))
        elif isinstance(t, Parenthesis):
            res.append(parse_parenthesis(t))
        elif isinstance(t, IdentifierList):
            res.append(parse_identifier_list(t))
        else:
            res.append(parse_components(t)) if parse_components(t) else res
    return res


def parse_operation(operation):
    assert isinstance(operation, Operation)
    res = []
    for t in operation.tokens:
        if isinstance(t, Identifier):
            res.append(t)
        elif isinstance(t, Parenthesis):
            res.append(parse_parenthesis(t))
        elif isinstance(t, IdentifierList):
            res.append(parse_identifier_list(t))
        else:
            res.append(parse_components(t)) if parse_components(t) else res
    return res


def parse_where(where):
    res = []
    for t in where.tokens:
        if isinstance(t, Comparison):
            res.append(parse_comparison(t))
        else:
            res.append(parse_components(t)) if parse_components(t) else res
    return res


def parse_comparison(comparison):
    # assert isinstance(comparison, Comparison)
    res = []
    for t in comparison.tokens:
        if isinstance(t, Identifier):
            res.append(t)
        elif isinstance(t, Parenthesis):
            res.append(parse_parenthesis(t))
        elif isinstance(t, IdentifierList):
            res.append(parse_identifier_list(t))
        else:
            res.append(parse_components(t)) if parse_components(t) else res
    return res


def parse_function(function):
    assert isinstance(function, Function)
    res = [function.get_name(), []]
    parameters = function.get_parameters()
    for t in parameters:
        if isinstance(t, Identifier):
            res[1].append(t)
        elif isinstance(t, Parenthesis):
            res[1].append(parse_parenthesis(t))
        elif isinstance(t, IdentifierList):
            res[1].append(parse_identifier_list(t))
        else:
            res[1].append(parse_components(t)) if parse_components(t) else res
    return res


def parse_components(parsed, num_columns=set()):
    all_types = [
        "Text",
        "Whitespace",
        "Newline",
        "Punctuation",
        "Keyword",
        "DML",
        "Wildcard",
        "Operator",
        "Literal",
        "String",
        "Name",
    ]
    ignore_types = [
        T.Text.Whitespace,
        T.Text.Whitespace.Newline,
        T.Punctuation,
        T.Keyword.DML,
        T.Wildcard,
    ]
    if isinstance(parsed, Token) and parsed.ttype and parsed.ttype[0] in all_types:
        if parsed.ttype not in ignore_types:
            return parsed
    else:
        res = []
        """
        print(
            parsed.value,
            type(parsed),
            parsed.ttype[0] if parsed.ttype else parsed.ttype,
        )
        """
        for t in parsed:
            if isinstance(t, Identifier):
                res.append(t)
            if isinstance(t, IdentifierList):
                res.append(parse_identifier_list(t))
            elif isinstance(t, Parenthesis):
                res.append(parse_parenthesis(t))
            elif isinstance(t, Operation):
                res.append(parse_operation(t))
            elif isinstance(t, Where):
                res.append(parse_where(t))
            elif isinstance(t, Function):
                res.append(parse_function(t))
            # @TODO: add other types
            else:
                # raise ValueError('Unexpected token: {}|{}'.format(t.value, type(t)), )
                if isinstance(t, Token) and t.ttype and t.ttype[0] in all_types:
                    if t.ttype not in ignore_types:
                        res.append(t)
        return res


def _get_all_columns(preparsed):
    res = []
    for p in preparsed:
        if isinstance(p, list):
            get_all_columns(p)
        else:
            if isinstance(p, Identifier):
                _type = "unknown"
                if "max" in preparsed or "min" in preparsed or "avg" in preparsed:
                    _type = "NUMERIC"
                _type_list = [
                    p.ttype[0]
                    if (not isinstance(p, list) and p.ttype is not None)
                    else "list"
                    for p in preparsed
                ]
                if "Operator" in _type_list:
                    _type = "NUMERIC"
                res.append((p.value, _type))
        return res
