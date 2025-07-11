#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:56:27 2024

@author: cjymain
"""

import duckdb
import traceback

from technically.utils.log import get_logger
from technically.utils.exceptions import DuckDBCloseError


class DuckDB:
    "Establishes DuckDB connection and facilitates actions to it."

    def __init__(self, db_path, mode="single", read_only=False):
        """
        Parameters
        ----------
        db_path : Full file path to existing/desired database location.
        mode (single): Defaults to single DB connection. Use "multi" to connect multiple DBs.
        log : Full file path to existing/desired log location.
        read_only (False): Whether to open database in read only mode
        existing_con (None): Pass an existing DuckDB connection object
            outside of context block to access class functions
        """
        self.mode = mode

        if mode == "single":
            self.db_path = db_path
            self.read_only = read_only
        elif mode == "multi":
            self.db_paths = db_path
            self.read_only_lst = read_only
            self.cons = {}

    def __enter__(self):
        # Catch Exceptions
        self.binder_err = duckdb.BinderException
        self.constraint_err = duckdb.ConstraintException
        self.invalid_input = duckdb.InvalidInputException
        self.already_exists = duckdb.CatalogException

        if self.mode == "single":
            self.con = duckdb.connect(database=self.db_path, read_only=self.read_only)
            self.sql = self.con.sql
            self.execute = self.con.execute
            return self
        elif self.mode == "multi":
            for path, read_only in zip(self.db_paths, self.read_only_lst):
                # I.E. /home/user/mydb.duck --> mydb
                db_name = path.split("/")[-1].split(".")[0]
                self.cons[db_name] = duckdb.connect(database=path, read_only=read_only)
            # Creates shorthands for these methods --> db.sql['prices']...
            self.sql = {db_name: con.sql for db_name, con in self.cons.items()}
            self.execute = {db_name: con.execute for db_name, con in self.cons.items()}

            return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        try:
            if exc_type is not None:
                error_code = "".join(
                    traceback.format_exception(
                        exc_type, exc_value, exc_traceback
                    )
                )
                get_logger().error(error_code)
        except Exception as e:
            raise DuckDBCloseError()
        finally:
            if self.mode == "single":
                self.con.close()
            elif self.mode == "multi":
                for db in self.cons.values():
                    db.close()

    def close(self, db_name):
        "Closes a specific DB connection in multi mode."
        if self.mode == "multi":
            self.cons[db_name].close()
        else:
            raise Exception("Cannot close single DB connection.")
        return

    def _get_db_name(self, db_name):
        "Assists in identifying correct DB to use in a function while in multi mode."
        if self.mode == "single":
            return self.con
        elif self.mode == "multi":
            if db_name is None:
                get_logger().error(DBNotSpecifiedError())
            return self.cons[db_name]
    
    def has_table(self, table_name: str, db_name=None):
        con = self._get_db_name(db_name)

        result = con.execute('''
            SELECT 
                * 
            FROM 
                information_schema.tables
            WHERE 
                table_name = ?;
            ''', [table_name]
        ).fetchall()

        return len(result) > 0
    
    def table_info(self, table_name: str, db_name=None):
        """
        Parameters
        ----------
        table : str
            Name of SQL table.

        Returns
        -------
        Dictionary: {"col_name": "col_dtype"...}
        """
        con = self._get_db_name(db_name)

        result = con.execute(f'''
            PRAGMA table_info("{table_name}");
            '''
        ).fetchall()
        table_dict = {col[1]: col[2] for col in result}

        return table_dict
    
    def add_missing_columns_to_table(self, table_name: str, columns: list, db_name=None):
        """
        :param table_name:
        :param columns: Keys are column names, values are column dtypes
        :param db_name:
        :return:
        """
        con = self._get_db_name(db_name)

        # Retreive current existing columns
        existing_cols = con.execute('''
            SELECT 
                column_name 
            FROM 
                information_schema.columns 
            WHERE 
                table_name = ?;
            ''', [table_name]
        ).fetchall()
        existing_cols = [col[0] for col in existing_cols]

        columns_to_add = [col for col in columns if col not in existing_cols]

        for column in columns_to_add:
            con.execute(f'''
                ALTER TABLE "{table_name}"
                    ADD COLUMN "{column}" DOUBLE;
                '''
            )


    