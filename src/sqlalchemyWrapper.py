"""SQLAlchemy wrapper around a database."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, List, Optional

from sqlalchemy import MetaData, create_engine, inspect, select, text, func
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
import random
random.seed(123)


class SQLDatabase:
    """SQLAlchemy wrapper around a database."""
    def __init__(
        self,
        engine: Engine,
        schema: Optional[str] = None,
        metadata: Optional[MetaData] = None,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 3,
        custom_table_info: Optional[dict] = None,
        only_bind_include_tables: bool = False,
        low_cardinality_threshold: int = 10,
        indexes_in_table_info: bool = False,
        lang=None,
    ):
        """Create engine from database URI."""
        self._engine = engine
        self._schema = schema
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")
        if only_bind_include_tables and not include_tables:
            raise ValueError("Must specify which tables to bind")

        self._inspector = inspect(self._engine)
        self._all_tables = set(self._inspector.get_table_names(schema=schema))
        self._include_tables = set(include_tables) if include_tables else set()
        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"include_tables {missing_tables} not found in database"
                )
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"ignore_tables {missing_tables} not found in database"
                )

        if not isinstance(sample_rows_in_table_info, int):
            raise TypeError("sample_rows_in_table_info must be an integer")

        self._sample_rows_in_table_info = sample_rows_in_table_info
       
        self._low_cardinality_threshold = low_cardinality_threshold
        self._indexes_in_table_info = indexes_in_table_info
        self._custom_table_info = custom_table_info
        self._lang=lang

        if self._custom_table_info:
            if not isinstance(self._custom_table_info, dict):
                raise TypeError(
                    "table_info must be a dictionary with table names as keys and the "
                    "desired table info as values"
                )
            # only keep the tables that are also present in the database
            intersection = set(self._custom_table_info).intersection(self._all_tables)
            self._custom_table_info = dict(
                (table, self._custom_table_info[table])
                for table in self._custom_table_info
                if table in intersection
            )

        self._metadata = metadata or MetaData()
        tables_to_bind = (
            self._include_tables if only_bind_include_tables else self._all_tables
        )
        self._metadata.reflect(
            bind=self._engine, schema=self._schema, only=tables_to_bind
        )



    @classmethod
    def from_uri(cls, database_uri: str, **kwargs: Any) -> SQLDatabase:
        """Construct a SQLAlchemy engine from URI."""
        return cls(create_engine(database_uri), **kwargs)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self._engine.dialect.name

    def get_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return self._include_tables
        return self._all_tables - self._ignore_tables

    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        return self.get_table_info()

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        all_table_names = self.get_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
               and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        for table in meta_tables:
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
                continue

            # add create table command
            create_table = str(CreateTable(table).compile(self._engine))

            if self._sample_rows_in_table_info:
                # build the select command
                command = select(table).limit(self._sample_rows_in_table_info)

                # save the command in string format
                select_star = (
                    f"SELECT * FROM '{table.name}' LIMIT "
                    f"{self._sample_rows_in_table_info}"
                )

                # save the columns in string format
                columns_str = "\t".join([col.name for col in table.columns])

                # get the sample rows
                try:
                    with self._engine.connect() as connection:
                        try:
                            sample_rows = connection.execute(command)
                            # shorten values in the sample rows

                            sample_rows = list(
                                map(lambda ls: [str(i)[:100]
                                                for i in ls], sample_rows)
                            )
                        except TypeError as e:
                            # print("***Back to literal querying...***")
                            sample_rows = connection.exec_driver_sql(select_star)
                            # shorten values in the sample rows
                            sample_rows = list(
                                map(lambda ls: [str(i)[:100]
                                                for i in ls], sample_rows)
                            )

                    # save the sample rows in string format
                    sample_rows_str = "\n".join(
                        ["\t".join(row) for row in sample_rows])
                except ProgrammingError:
                    sample_rows_str = ""
                # in some dialects when there are no rows in the table a
                # 'ProgrammingError' is returned

                # build final info for table
                tables.append(
                    create_table
                    + select_star
                    + ";\n"
                    + columns_str
                    + "\n"
                    + sample_rows_str
                )

            else:
                tables.append(create_table)

        final_str = "\n\n".join(tables)
        return final_str

    def get_table_info_dict(self, table_names: Optional[List[str]] = None) -> dict:
        all_table_names = self.get_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(
                    f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
               and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []

        for table in meta_tables:
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
            else:
                tables.append(table)
        tables_dict = {}
        for table in tables:
            cols = []
            col_details = []
            pk = []
            fks = []
            sample_rows = self.get_tbl_samples_dict(table)
            num_rows = self.get_rows_of_a_table(table)
            for col in table.columns:
                # distinct_values = self.count_distinct_values_of_a_col(table, col)
                distinct_values = list(range(10)) # to solve the error
                # cardinality = len(distinct_values) / num_rows
                cols.append([col.name, str(col.type).split('.')[-1]])
                if col.primary_key:
                    pk.append(col.name)
                if len(col.foreign_keys) > 0:
                    for fk in list(col.foreign_keys):
                        fks.append([f'{table.name}.{col.name}', fk.target_fullname])
                # here we use 3 simple conditions to filterout the categorical values:
                # 1. cardinality < 0.3
                # 2. total len(distinct_values) < 20
                # 3. '_id' not in name or name is not equal to 'id'
                # if cardinality < 0.5 and len(distinct_values) < 20 and (
                #         '_id' not in col.name.lower() or col.name.lower() == 'id'):  # maybe a categorical value
                #     col_details.append(
                #         {'is_categorical': True, 'cardinality': cardinality, 'distinct_values': distinct_values})
                # else:
                #     col_details.append(
                #         {'is_categorical': False, 'cardinality': cardinality, 'distinct_values': distinct_values[:20]})
            tables_dict[table.name] = {
                'COL': cols,
                'COL_DETAILS': col_details,
                'PK': pk,
                'FK': fks,
                'sample_rows': sample_rows
            }
        return tables_dict

    def get_table_info_with_cardinality(self, table_names: Optional[List[str]] = None) -> str:
        """
        Get information about specified tables.
        """
        all_table_names = self.get_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
            and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        
        for table in meta_tables:
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
                continue

            # add create table command
            create_table = str(CreateTable(table).compile(self._engine))
            table_info = f"{create_table.rstrip()}"
            if self._indexes_in_table_info:
                table_info += f"\n{self._get_table_indexes(table)}\n"
            if self._sample_rows_in_table_info:
                self._get_sample_rows_and_metadata(table)
                try:
                    table_info += f"\n{self._get_sample_rows_and_metadata(table)}\n"
                except Exception:
                    pass
            tables.append(table_info)
        
        tables.sort()
        final_str = "\n\n".join(tables)
        return final_str
    
    def get_table_info_rows(self, table_names: Optional[List[str]] = None) -> str:
        """
        Get information about specified tables.
        """
        all_table_names = self.get_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
            and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        
        for table in meta_tables:
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
                continue

          
            table_info = ""
            try:
                table_info += f"\n{self._get_rows(table)}\n"
            except Exception:
                pass
            tables.append(table_info)
        
        tables.sort()
        final_str = "\n\n".join(tables)
        return final_str
    
    def _get_table_indexes(self, table: Table) -> str:
        indexes = self._inspector.get_indexes(table.name)
        indexes_formatted = "\n".join(map(_format_index, indexes))
        return f"Table Indexes:\n{indexes_formatted}"
    
    def _get_sample_rows(self, table: Table) -> str:
        limiting_factor = 200
        # build the select command
        command = select(table).limit(limiting_factor)
        
        try:
            with self._engine.connect() as connection:
                response = ""
                sample_rows_result = connection.execute(command)
                sample_rows = sample_rows_result.fetchall()

                 # Create sections for high and low cardinality columns
                high_cardinality_section = f"/*\nColumns in {table.name} and {str(self._sample_rows_in_table_info)} examples in each column for high cardinality columns :"  # noqa: E501
                low_cardinality_section = f"/*\nColumns in {table.name} and all categories for low cardinality columns :"  # noqa: E501

                low_columns = ""
                high_columns = ""
                
                for column, index in zip(table.columns,range(len(table.columns))):
                    column_name = column.name
                    values = [str(row[index]) for row in sample_rows]

                    # Determine if the column is high or low cardinality based on the threshold  # noqa: E501
                    unique_values = set(values)
                    if len(unique_values) > self._low_cardinality_threshold:
                        high_columns += f"\n{column_name} : {', '.join(list(unique_values)[:self._sample_rows_in_table_info])}"  # noqa: E501
                    else:
                        low_columns += f"\n{column_name} : {', '.join(unique_values)}"  # noqa: E501

                if high_columns:
                    high_cardinality_section += high_columns + "\n*/\n"
                    response += high_cardinality_section

                if low_columns:
                    low_cardinality_section += low_columns + "\n*/"
                    response += low_cardinality_section

    

        except ProgrammingError:
                response = ""

        return response
    
    def _get_sample_rows_and_metadata(self, table: Table) -> str:
        limiting_factor = 200
        # build the select command
        command = select(table).limit(limiting_factor)

        
        if self._lang=='en':
            comand_meta_data_column=  (
                        f"SELECT name,title_en, example_values FROM metadata_table_columns "
                        f"WHERE table_name ='{table.name}'"
                    )
            comand_meta_data_table = (
                        f"SELECT name, description FROM metadata_tables "
                        f"WHERE name ='{table.name}'")
        else:
            
            comand_meta_data_column=  (
                        f"SELECT name,title, example_values FROM metadata_table_columns "
                        f"WHERE table_name ='{table.name}'"
                        
                    )
            comand_meta_data_table = (
                        f"SELECT name, description FROM metadata_tables "
                        f"WHERE name ='{table.name}'")
        
        meta_data=""
       
        with self._engine.begin() as connection:
            if table.name !='spatial_unit' :
                if self._schema is not None:
                    connection.exec_driver_sql(
                        f"SET search_path TO {self._schema}")
                cursor = connection.execute(text(comand_meta_data_column))
                meta_column = cursor.fetchall()
               
                cursor_table = connection.execute(text(comand_meta_data_table))
                meta_table = cursor_table.fetchall()
               
                
                
                
               # meta_data += f"\n/*\n{meta_table[-1][0]}: {meta_table[-1][1]}\n*/" #if self._lang=='en' else f"{item[0]}, {item[1]}, {item[2]}\n"
                
                meta_data += "\n/*\nColumn name, Column discription, Example values\n" #if self._lang=='en' else "\n/*\nColumn name, Column discription, Column discription in English\n"
                for item in meta_column:
                    meta_data += f"{item[0]}, {item[1]}, {item[2]}\n" #if self._lang=='en' else f"{item[0]}, {item[1]}, {item[2]}\n"
        try:
            # print("***Back to literal querying...***")
            with self._engine.connect() as connection:
              
                response = ""
                sample_rows_result = connection.execute(command)
                sample_rows = sample_rows_result.fetchall()

                 # Create sections for high and low cardinality columns
                high_cardinality_section = f"/*\nColumns in {table.name} and {str(self._sample_rows_in_table_info)} examples in each column for high cardinality columns :"  # noqa: E501
                low_cardinality_section = f"/*\nColumns in {table.name} and all categories for low cardinality columns :"  # noqa: E501

                low_columns = ""
                high_columns = ""
                
                for column, index in zip(table.columns,range(len(table.columns))):
                    column_name = column.name
                    values = [str(row[index]) for row in sample_rows]

                    # Determine if the column is high or low cardinality based on the threshold  # noqa: E501
                    unique_values = set(values)
                    if len(unique_values) > self._low_cardinality_threshold:
                        high_columns += f"\n{column_name} : {', '.join(list(unique_values)[:self._sample_rows_in_table_info])}"  # noqa: E501
                    else:
                        low_columns += f"\n{column_name} : {', '.join(unique_values)}"  # noqa: E501

                if high_columns:
                    high_cardinality_section += high_columns + "\n*/\n"
                    response += high_cardinality_section

                # if low_columns:
                #     low_cardinality_section += low_columns + "\n*/"
                #     response += low_cardinality_section
                if meta_data!="":
                    response += meta_data+"*/" 

        except ProgrammingError:
                response = ""
        
        return response 
    

    def _get_rows(self, table: Table) -> str:
        # build the select command
        command = select(table)

        command=  (
                        f"SELECT * FROM {table.name} "
                        f"WHERE year=2021 and spatialunit_uid='0_CH'"
                    )

        if self._lang=='en':
            comand_meta_data_column=  (
                        f"SELECT name,title_en, example_values FROM metadata_table_columns "
                        f"WHERE table_name ='{table.name}'"
                    )
            comand_meta_data_table = (
                        f"SELECT name, description FROM metadata_tables "
                        f"WHERE name ='{table.name}'")
        else:
            
            comand_meta_data_column= (
                        f"SELECT name,title FROM metadata_table_columns "
                        f"WHERE table_name ='{table.name}'"
                        
                    )
            comand_meta_data_table = (
                        f"SELECT name, description FROM metadata_tables "
                        f"WHERE name ='{table.name}'")
        
        meta_data=""
       
        with self._engine.begin() as connection:
            if table.name !='spatial_unit' :
                if self._schema is not None:
                    connection.exec_driver_sql(
                        f"SET search_path TO {self._schema}")
                cursor = connection.execute(text(comand_meta_data_column))
                meta_column = cursor.fetchall()
               
                cursor_table = connection.execute(text(comand_meta_data_table))
                meta_table = cursor_table.fetchall()
               
                
               # meta_data += f"\n/*\n{meta_table[-1][0]}: {meta_table[-1][1]}\n*/" #if self._lang=='en' else f"{item[0]}, {item[1]}, {item[2]}\n"
                
                meta_data += "\nColumn name, Column discription\n" #if self._lang=='en' else "\n/*\nColumn name, Column discription, Column discription in English\n"
                for item in meta_column:
                    meta_data += f"{item[0]}, {item[1]}\n" #if self._lang=='en' else f"{item[0]}, {item[1]}, {item[2]}\n"
        try:
            # print("***Back to literal querying...***")
            with self._engine.connect() as connection:
              
                response = ""
                rows_result = connection.execute(command)
                rows = rows_result.fetchall()

                 # Create sections for high and low cardinality columns
                table_section = f"\nColumns in {table.name} and values in each column:"  # noqa: E501

                columns_value = ""
               
                response+='|'.join([column.name for column in table.columns])+'\n'
                response+='|'.join(["----" for x in range(len(table.columns))])+'\n'
                for row in rows:
                    values = [str(row[index]) for index in range(len(table.columns))]
                    response+='|'.join(values)+'\n'
            
            if meta_data!="":
                    response = meta_data +"\n"+ response

        except ProgrammingError:
                response = ""

        return response 
    

    def run_scipt(self, command) -> str:
        # build the select command
        try:
            # print("***Back to literal querying...***")
            with self._engine.connect() as connection:
                response = ""
                rows_result = connection.execute(command)
                columns=rows_result.keys()
                rows = rows_result.fetchall()

                response+='|'.join([column for column in columns])+'\n'
                response+='|'.join(["----" for x in range(len(columns))])+'\n'
                for row in rows:
                    values = [str(row[index]) for index in range(len(columns))]
                    response+='|'.join(values)+'\n'
                
                

        except ProgrammingError:
                response = ""

        return response 
    def get_tbl_samples_dict(self, table):
        sample_rows_dict = {}
        if self._sample_rows_in_table_info:
            # build the select command
            command = select(table).limit(self._sample_rows_in_table_info)

            # save the command in string format
            select_star = (
                f"SELECT * FROM '{table.name}' LIMIT "
                f"{self._sample_rows_in_table_info}"
            )

            # save the columns
            columns = [col.name for col in table.columns]

            # get the sample rows
            try:
                with self._engine.connect() as connection:
                    try:
                        sample_rows = connection.execute(command)
                        # shorten values in the sample rows

                        sample_rows = list(
                            map(lambda ls: [str(i)[:100]
                                            for i in ls], sample_rows)
                        )
                    except TypeError as e:
                        # print("***Back to literal querying...***")
                        sample_rows = connection.exec_driver_sql(select_star)
                        # shorten values in the sample rows
                        sample_rows = list(
                            map(lambda ls: [str(i)[:100]
                                            for i in ls], sample_rows)
                        )
                sample_rows_T = list(map(list, zip(*sample_rows)))
                for col, rows in zip(columns, sample_rows_T):
                    sample_rows_dict[col] = rows
            except ProgrammingError as e:
                print(f'Warning: sampling error:{table.name},{str(e)}')
                sample_rows_dict = {}
        return sample_rows_dict

    def get_rows_of_a_table(self, table):
        command = select(func.count()).select_from(table)
        try:
            with self._engine.connect() as connection:
                num_rows = connection.execute(command)
                # print(table.name)
                return num_rows.scalar()
        except ProgrammingError:
            print('Warning: categorical error')
            return None

    def count_distinct_values_of_a_col(self, table, column, num_limit=100):
        command = select(func.count(column), column).group_by(column).order_by(func.count(column).desc()).limit(
            num_limit)
        try:
            with self._engine.connect() as connection:
                sample_rows = connection.execute(command).fetchall()
                # print(table.name, column.name)
                return [list(r) for r in sample_rows]
        except ProgrammingError:
            print('Warning: categorical error')
            return []

    def run(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self._engine.begin() as connection:
            if self._schema is not None:
                connection.exec_driver_sql(
                    f"SET search_path TO {self._schema}")
            cursor = connection.execute(text(command))
            if cursor.returns_rows:
                if fetch == "all":
                    result = cursor.fetchall()
                elif fetch == "one":
                    result = cursor.fetchone()[0]
                else:
                    raise ValueError(
                        "Fetch parameter must be either 'one' or 'all'")
                return str(result)
        return ""

    def get_table_info_no_throw(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        try:
            return self.get_table_info(table_names)
        except ValueError as e:
            """Format the error message"""
            return f"Error: {e}"

    def run_no_throw(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.

        If the statement throws an error, the error message is returned.
        """
        try:
            return self.run(command, fetch)
        except SQLAlchemyError as e:
            """Format the error message"""
            return f"Error: {e}"

    def dict2str(self, d):
        text = []
        for t, v in d.items():
            _tbl = f'{t}:'
            cols = []
            pks = ['PK:']
            fks = ['FK:']
            for col in v['COL']:
                cols.append(f'{col[0]}:{self.aliastype(col[1])}')
            for pk in v['PK']:
                pks.append(pk)
            for fk in v['FK']:
                fks.append('='.join(list(fk)))

            tbl = '\n'.join([_tbl, ', '.join(cols), ' '.join(pks), ' '.join(fks)])
            text.append(tbl)
        return '\n'.join(text)

    def aliastype(self, t):
        _t = t[:3].lower()
        if _t in ['int', 'tin', 'sma', 'med', 'big', 'uns', 'rea', 'dou', 'num', 'dec']:
            res = 'N'  # numerical value
        elif _t in ['tex', 'var', 'cha', 'nch', 'nat', 'nva', 'clo']:
            res = 'T'  # text value
        elif _t in ['boo']:
            res = 'B'
        elif _t in ['dat']:
            res = 'D'
        else:
            raise ValueError('Unsupported data type')
        return res

def aliastype(t):
    _t = t[:3].lower()
    if _t in ['int', 'tin', 'sma', 'med', 'big', 'uns', 'rea', 'dou', 'num', 'dec','flo']:
        res = 'N'  # numerical value
    elif _t in ['tex', 'var', 'cha', 'nch', 'nat', 'nva', 'clo','tim']:
        res = 'T'  # text value
    elif _t in ['boo']:
        res = 'B'
    elif _t in ['dat']:
        res = 'D'
    else:
        print(t)
        raise ValueError('Unsupported data type')
    return res


def formatting(ddl, alias=False):

    schema_prompt = ""
    for k, v in ddl.items():
        # print(k)
        schema_prompt += f"[Table]: {k}" + "\n[Column names, Type]:\n"
        for l in v['COL']:
            # print(f"{l[0]}, {l[-1]}" )
            schema_prompt += f"{l[0]}, {aliastype(l[-1]) if alias else l[-1] }" + "\n"
        # print("PKs:",",". join(v['PK']))
        schema_prompt += "\n[PKs]: " + ", ".join(v['PK']) + "\n"
        if v['FK'] != []:
            fk_string = []
            for fk in v['FK']:
                fk_string.append(f"{fk[0].strip()},{fk[-1]}")
            # print("FKs:", "; ".join(fk_string))
            schema_prompt += "\n[FKs]: " + "; ".join(fk_string) + "\n"
        else:
            schema_prompt += '\n[FKs]: "" '
        # print("Rows")
        if len(v['sample_rows']) > 0:
            schema_prompt += "\n[Sample rows]:\n"
            for column,values in v['sample_rows'].items():
                # print(",".join(row))
                schema_prompt += column+":\t" + "\t".join(values) + "\n"
        schema_prompt += "\n"
    return schema_prompt



def schema_db_postgres_statbot(db_config, include_tables=None, sample_number=0, alias=False,lang=None):
    host = db_config['host']
    port = db_config['port']
    database = db_config['database']
    username = db_config['username']
    password = db_config['password']
    schema = db_config['schema']
    database_uri = f'postgresql://{username}:{password}@{host}:{str(port)}/{database}'
    db = SQLDatabase.from_uri(database_uri, schema=schema,
                              include_tables= include_tables,
                              sample_rows_in_table_info=sample_number,lang=lang)

    return  db.get_table_info_with_cardinality()
