#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:18:37 2023

@author: cjymain
"""

from sqlalchemy import (
    create_engine,
    orm,
    text,
    MetaData,
    Table,
    schema,
    Column,
    Index,
    types,
    inspect,
    update,
    exc
)
from contextlib import contextmanager
import traceback


class Database:
    "Establishes SQL DB connection and facilitates queries to it."

    def __init__(self, db_path, log, db_type="duckdb", multithreaded=False):
        """
        Parameters
        ----------
        db_path : Full file path to existing/desired database location.
        log : Full file path to existing/desired log location.

        """
        self.db_path = db_path
        self.log = log
        self.db_type = db_type
        self.multithreaded = multithreaded
        self.conn = None

    def __enter__(self):
        self.engine = create_engine(f"{self.db_type}:///{self.db_path}", echo=False)
        session_factory = orm.sessionmaker(bind=self.engine)
        if self.multithreaded:
            self.Session = orm.scoped_session(session_factory)
            self.cursor = self.Session()
        else:
            self.cursor = session_factory()
            
        self.meta = MetaData
        self.table = Table
        self.create_table = schema.CreateTable
        self.col = Column
        self.idx = Index
        self.dtype = types
        self.inspect = inspect
        self.update = update
        self.exc = exc
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        try:
            if exc_type is not None:
                error_code = "".join(
                    traceback.format_exception(
                        exc_type, exc_value, exc_traceback
                    )
                )
                self.log.error(error_code)
                self.cursor.rollback()
            else:
                self.cursor.commit()
        except Exception as e:
            self.log.error(f"Sqlalchemy session cleanup error: {str(e)}")
            raise
        finally:
            self.cursor.close()
            if self.multithreaded:
                self.Session.remove()
            self.engine.dispose()

    # Singular methods
    def query_to_lst(self, query: str, byRow=False):
        """
        If byRow=False, returns columns as their own, comma separated tuples.
        To isolate columns, call --> x, y = query("SELECT x, y FROM table;").

        If byRow=True, returns rows as their own, comma separated tuples.
        Best for quickly converting tables to DataFrames.
        """
        response = self.cursor.execute(text(query)).fetchall()
        if response == []:
            return
        elif byRow == True:
            response = [list(column) for column in zip(*response)]
        elif len(response[0]) == 1:
            response = [val[0] for val in response]
        return response
    
    def select_raw(self, query: str):
        return self.cursor.execute(text(query)).fetchall()

    def query_to_array(self, query: str, dtype: type):
        import numpy as np

        response = self.cursor.execute(text(query)).fetchall()
        if response == []:
            return
        else:
            response = np.array(response, dtype=dtype).T
        return response

    def query_to_df(self, query: str, index=None, dict=False):
        """
        Returns DataFrame by default; 
        will convert DataFrame to Dictionary
        if dict=True.

        """
        import pandas as pd
        df = pd.read_sql_query(query, con=self.engine, index_col=index)
        if dict:
            df = df.to_dict(orient="list")
        return df

    def write_query(self, query: str, commit=True):
        self.cursor.execute(text(query))
        if commit:
            self.cursor.commit()
        return

    def table_info(self, table: str):
        """
        Parameters
        ----------
        table : str
            Name of SQL table.

        Returns
        -------
        Dictionary: {"col_name": "col_dtype"...}
        """
        info = self.cursor.execute(
            text(f"PRAGMA table_info('{table}');")
        ).fetchall()
        info_dict = {col[1]: col[2] for col in info}
        return info_dict

    def create_index(self, table: str, columns: list, unique: bool):
        """
        Parameters
        ----------
        table : str
            Table to create index on.
        columns : list
            List of column names the index will include.
        unique : bool
            Whether the combination of index columns must be unique.
        """
        if unique:
            uq = "UNIQUE"
        else:
            uq = ""
        idx_name = "idx_" + "_".join(columns)
        formatted_columns = ", ".join(columns)
        self.cursor.execute(text(f"""
            CREATE {uq}
            INDEX {idx_name}
            ON "{table}" ({formatted_columns});
            """
        ))
        self.cursor.commit()
        return
    
    # Aggregate methods
    def sync_metadata_to_table_names(self):
        # Ticker table names
        established_names = self.query_to_lst("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table';
            """
        )
        # Ticker metadata names
        updated_names = self.query_to_lst("SELECT ticker FROM metadata;")

        tables_to_rename = []
        for name in updated_names:
            if name.endswith("_old"):
                base = name[:-4]
            else:
                base = name

            if base in established_names and base != name:
                tables_to_rename.append((base, name))

        # Reanmes all tables to reflect their active/inactive status in metadata
        for old_name, new_name in tables_to_rename:
            self.write_query(f"""
                ALTER TABLE "{old_name}" 
                RENAME TO "{new_name}";
                """
            )
            self.log.info(f"""{old_name} moved to {
                              new_name}. Ticker is now inactive.""")

    def delete_sql_tables(self, tables: list, by_row=False, alt_table=None, primary_col="ticker"):
        """
        by_row & table_name: If tickers are a column within a table as opposed to a full table, 
                            set by_row to True, alt_table to table that holds tickers as column, 
                            and primary_col to the column that holds tickers.

        """
        if len(tables) == 0:
            return
        for table in tables:
            if by_row == False:
                self.cursor.execute(f"""
                    DROP TABLE IF EXISTS "{table}";
                    """
                )
            else:
                self.cursor.execute(f"""
                    DELETE FROM "{alt_table}"
                    WHERE {primary_col} = '{table}';
                    """
                )

    def lst_sql_tables(self):
        table_names = self.cursor.execute(text("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name ASC; 
            """
        )).fetchall()
        table_names = [x[0] for x in table_names]

        return table_names

@contextmanager
def multi_db_connection(db_path1, db_path2, log, db_type1="duckdb", db_type2="duckdb"):
    with Database(db_path1, log, db_type1) as db1, Database(db_path2, log, db_type2) as db2:
        yield db1, db2
