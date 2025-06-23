class TCException(Exception):
    def __init__(self, message=""):
        super().__init__(message)

class NotEnoughDataError(TCException):
    def __init__(self, ticker, len):
        super().__init__(f'''Not enough data for {ticker} to do technical analysis. 
                         Has {len} rows, needs 120.''')

class DeadTickerError(TCException):
    def __init__(self, ticker):
        super().__init__(f'Minimal price activity for {ticker}. It will be discluded from analysis.')

class ParquetRemovalError(TCException):
    def __init__(self, ticker):
        super().__init__(f'Parquet removal failed for {ticker}.')

class NoDataReturnedError(TCException):
    def __init__(self, method_name, ticker):
        super().__init__(f'Data acquisition ({method_name}) for {ticker} returned no data.')

class DatabaseUpdateError(TCException):
    def __init__(self, ticker):
        super().__init__(f'Database update failed for {ticker}.')

class DuckDBCloseError(TCException):
    def __init__(self):
        super().__init__(f'DuckDB connection close failed.')

class PySparkCloseError(TCException):
    def __init__(self):
        super().__init__(f'PySpark connection close failed.')

class DBNotSpecifiedError(TCException):
    def __init__(self):
        super().__init__(f'''Database not specified for method operating on multi-db connection.
                         You must specify a db name when in multi-db mode.''')