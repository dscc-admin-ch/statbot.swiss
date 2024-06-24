import os
import re
import asyncio
import sqlite3
import threading
from typing import Tuple, Any, List, Set
from decimal import Decimal
from itertools import product
from collections import defaultdict
import random
import time
import pickle as pkl
import subprocess
from itertools import chain
import datetime
from sql_analytics.src.utils.sql_database import SQLDatabase
from sqlalchemy.exc import ProgrammingError, DataError


def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


# unorder each row in the table
# [result_1 and result_2 has the same bag of unordered row]
# is a necessary condition of
# [result_1 and result_2 are equivalent in denotation]
def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    res = False
    details = ""
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        if s1 == s2:
            res = True
        else:
            details = (
                f"Results are not identical in ORDERED comparison\nGT:{s1}\nGT{s2}"
            )

    else:
        if set(s1) == set(s2):
            res = True
        else:
            details = (
                f"Results are not identical in UNORDERED comparison\nGT:{s1}\nGT{s2}"
            )

    return res, details


# return whether two bag of relations are equivalent
def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    # we sample 20 rows and constrain the space of permutations
    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


# convert all Decimal values to float
def format_result(result: List[Tuple]) -> List[Tuple]:
    return [
        tuple(float(e) if isinstance(e, Decimal) else e for e in row) for row in result
    ]


# check whether two denotations are correct
def result_eq(
    result1: List[Tuple],
    result2: List[Tuple],
    order_matters: bool,
    is_hard=True,
    is_partial=False,
) -> (bool, str):
    result1 = format_result(result1)
    result2 = format_result(result2)
    details = "\n"
    if len(result1) == 0 and len(result2) == 0:
        return True, details

    # if length is not the same, then they are definitely different bag of rows
    if len(result1) != len(result2):
        return False, details

    num_cols = len(result1[0])

    # for soft it changes
    # if the results do not have the same number of columns, they are different
    if is_hard and len(result2[0]) != num_cols:
        details += f"results do not have the same number of columns, they are different, GT: {num_cols}, PT: {result2[0]}"
        return False, details

    # unorder each row and compare whether the denotation is the same
    # this can already find most pair of denotations that are different
    if is_hard and not quick_rej(result1, result2, order_matters):
        return False, details

    if is_hard:
        # the rest of the problem is in fact more complicated than one might think
        # we want to find a permutation of column order and a permutation of row order,
        # s.t. result_1 is the same as result_2
        # we return true if we can find such column & row permutations
        # and false if we cannot
        tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

        # on a high level, we enumerate all possible column permutations that might make result_1 == result_2
        # we decrease the size of the column permutation space by the function get_constraint_permutation
        # if one of the permutation make result_1, result_2 equivalent, then they are equivalent
        for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
            if len(perm) != len(set(perm)):
                continue
            if num_cols == 1:
                result2_perm = result2
            else:
                result2_perm = [permute_tuple(element, perm) for element in result2]
            if order_matters:
                if result1 == result2_perm:
                    details += f"ORDERED: FOUND matched permutation\nGT: {result1}\nPT: {result2}\nPT_perm: {result2_perm}"
                    return True, details
            else:
                # in fact the first condition must hold if the second condition holds
                # but the first is way more efficient implementation-wise
                # and we use it to quickly reject impossible candidates
                if set(result1) == set(result2_perm) and multiset_eq(
                    result1, result2_perm
                ):
                    details += f"UNORDERED: FOUND matched permutation\nGT: {set(result1)}\nPT: {set(result2)}\nPT_perm: {set(result2_perm)}"
                    return True, details

        if order_matters:
            details += f"Results are not identical in ORDERED comparison\nGT:{tab1_sets_by_columns}\nGT{result2}"
        else:
            details += f"Results are not identical in UNORDERED comparison\nGT:{tab1_sets_by_columns}\nGT{result2}"
        return False, details

    # for soft cases, we only check whether result_2 has all columns in result_1, even it might have more columns return
    # we can columnize the result_1 and result_2, and check whether result_2 has all columns in result_1
    hard_res = result_eq(result1, result2, order_matters, is_hard=True)
    _hard_res = hard_res[0]
    if _hard_res:
        return hard_res
    else:
        columnized_result_1 = list(zip(*result1))
        columnized_result_2 = list(zip(*result2))
        if not is_partial:
            if order_matters:
                for i, col in enumerate(columnized_result_1, start=1):
                    if len(col) > 0:
                        if col not in columnized_result_2:
                            details += f"Results are missing in SOFT comparison\nGT:{columnized_result_1[i]} is not in \nGT{columnized_result_2}"
                            return False, details
                details += f"Results are identical in SOFT comparison\nGT:{columnized_result_1}\nGT{columnized_result_2}"
                return True, details
            else:
                for i, col in enumerate(columnized_result_1, start=1):
                    if len(col) > 0:
                        if set(col) not in [set(c) for c in columnized_result_2]:
                            details += f"Results are missing in SOFT comparison\nGT:{columnized_result_1} is not in \nGT{columnized_result_2}"
                            return False, details
                details += f"Results are identical in SOFT comparison\nGT:{columnized_result_1}\nGT{columnized_result_2}"
                return True, details
        _soft_res = result_eq(result1, result2, order_matters, is_hard=False, is_partial=False)
        if _soft_res[0]:
            return _soft_res
        for i, col in enumerate(columnized_result_2, start=1):
            if len(col) > 0:
                if set(col) not in [set(c) for c in columnized_result_1]:
                    details += f"Results are missing in partial comparison\nGT:{columnized_result_2} is not in \nGT{columnized_result_1}"
                    return False, details
        details += f"Results are partially identical in comparison\nGT:{columnized_result_1}\nGT{columnized_result_2}"
        return True, details


def replace_cur_year(query: str, cur_year=None) -> str:
    if cur_year is None:
        cur_year = datetime.datetime.now().year
    return re.sub(
        "YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", cur_year, query, flags=re.IGNORECASE
    )


def sql_eq(
    sqldb,
    sql_1,
    sql_2,
    order_matters=True,
    with_details=False,
    fetch="many",
    max_rows=50,
    is_hard=True,
    is_partial=False,
):
    try:
        result1 = sqldb.run(sql_1, fmt="list", fetch=fetch, limit_num=max_rows)
    except (ProgrammingError, DataError) as e:
        result1 = ["error", "res_1", e._sql_message()]
    try:
        result2 = sqldb.run(sql_2, fmt="list", fetch=fetch, limit_num=max_rows)
    except (ProgrammingError, DataError) as e:
        result2 = ["error", "res_2", e._sql_message()]
    res, details = result_eq(
        result1,
        result2,
        order_matters=order_matters,
        is_hard=is_hard,
        is_partial=is_partial,
    )
    if with_details:
        return res, details, result1, result2
    return res, result1, result2


def test_world_cup():
    host = "testbed.inode.igd.fraunhofer.de"
    port = 18001
    database = "world_cup"
    username = "inode_readonly"
    password = "W8BYqhSemzyZ64YD"
    database_uri = f"postgresql://{username}:{password}@{host}:{str(port)}/{database}"
    sqldb = SQLDatabase.from_uri(database_uri)
    sqldb.schema = "exp_v3"
    # print(db.get_table_info_no_throw())
    # print(db.run('SELECT * FROM player LIMIT 5'))

    # Test case swapped
    sql_1 = """SELECT T2.winner, T1.teamname
FROM national_team AS T1
    JOIN world_cup_result AS T2 ON T1.team_id = T2.team_id
WHERE T2.winner = 'true' ORDER BY T2.year ASC LIMIT 5"""
    sql_2 = """SELECT T1.teamname, T2.winner
FROM national_team AS T1
    JOIN world_cup_result AS T2 ON T1.team_id = T2.team_id
WHERE T2.winner = 'true' ORDER BY T2.year ASC LIMIT 5"""
    print(*sql_eq(sqldb, sql_1, sql_2, order_matters=False, with_details=True))

from decimal import Decimal
def test_partial():
    
    gt_res = [(Decimal(2007), "Switzerland", "Foreign country", 896)]
    pred_res = [(2007, 896)]
    print(
        result_eq(gt_res, pred_res, order_matters=False, is_hard=True)
    )
    print(
        result_eq(gt_res, pred_res, order_matters=False, is_hard=False)
    )
    print(
        result_eq(gt_res, pred_res, order_matters=False, is_hard=False, is_partial=True)
    )


if __name__ == "__main__":
    test_partial()
