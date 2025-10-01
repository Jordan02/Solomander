import yfinance as yf
import pandas as pd
import os
import ccxt
import time
import json

try: 
    from .logger import log, stamp, pront
    from .utils import timedelta_to_str
except ImportError: 
    #for running as main script
    from logger import log, stamp, pront
    from utils import timedelta_to_str


def load_yfinance(symbol: str, start :str = "2023-01-01", end: str= "2023-12-31", interval: str ="1d") -> pd.DataFrame:
 
    """
    Download OHLCV data for a given ticker using yfinance.

    Args:
        ticker (str): Symbol, e.g. "AAPL" or "MNQ=F".
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
        interval (str): Candle interval. Default is '1d'.

    Returns:
        pd.DataFrame: Always returns a DataFrame (empty if no data).
    """
    # ----- CHECK IF THERE IS A VALID SYMBOL /META DATA FOR IT -----
    if load_symbol(symbol) is None:
        return pd.DataFrame()

    # ----- CHECK IF FILE EXISTS -----
    filename = f"yfin_{symbol}_{start}_{end}_{interval}.csv".replace("-", "").replace("/", "-")
    df = _load_data(filename) 
    if df is not None:
        stamp.success(f"✅ Data loaded from data/{filename} Opening now queen.")
        return df

    # ----- DOWNLOAD IF NOT -----
    try:
        df = yf.download(symbol, start=start, end=end, interval=interval)
    except Exception as e:
        log.error(f"❌ Error downloading from yfinance: {e}")
        return pd.DataFrame()

    # ----- FORMAT DATA -----

    df.index = pd.to_datetime(df.index) # convert index to datetime
    # remove second column name if it is a MultiIndex, and lowercase all column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        df.columns = df.columns.str.lower()
    else:
        df.columns = df.columns.str.lower()

    df = df.reindex(columns=["open", "close", "high", "low", "volume"]) #prefer OCHL order finplot

    # rename index
    df.columns.name = None
    df.reset_index(drop=True)
    df.index.name = "datetime"
    
    # ----- WRITE TO FILE -----

    _write_data(df,filename)
    stamp.success(f"✅ Data saved to data/{filename}. tz:{df.index.tz} {start} to {end} with {interval} interval")

    return df

def load_binance(symbol: str = "BTC/USDT", start: str = "2023-01-01", end: str = "2023-04-30", interval: str = "5m") -> pd.DataFrame:
    """
    Download OHLCV data for a given symbol from Binance using ccxt.

    Args:
        symbol (str): e.g. "BTC/USDT" or "ETH/USDT".
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
        interval (str): Candle interval. Default is '5m'.

    Returns:
        pd.DataFrame: Always returns a DataFrame (empty if no data).
    """

     # ----- check there is valid symbol data available -----
    if load_symbol(symbol) is None:
        return pd.DataFrame()

    # ----- CHECK IF FILE EXISTS -----
    filename = f"bin_{symbol}_{start}_{end}_{interval}.csv".replace("-", "").replace("/", "-")
    df = _load_data(filename) 
    if df is not None:
        stamp.success(f"✅ Data loaded from data/{filename} Opening now queen.")
        return df

    # ----- DOWNLOAD IF NOT -----

    # init binance
    exchange = ccxt.binance()
    since = exchange.parse8601(start + "T00:00:00Z")
    end_ts = exchange.parse8601(end + "T00:00:00Z")
    all_ohlcv = []
    limit = 1000  # Binance max per fetch

    while since < end_ts:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, since=since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # move past last timestamp
        time.sleep(exchange.rateLimit / 1000)  # be nice to the API

    if not all_ohlcv:
        log.error(f"❌ No data returned for {symbol}")
        return pd.DataFrame()

    # ----- DOWNLOAD IF NOT -----
    
    df = pd.DataFrame(all_ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df = df.reindex(columns=["open", "close", "high", "low", "volume"])  # reindex to OCHL order (your convention)

    # save
    _write_data(df,filename)
    stamp.success(f"✅ Data saved to data/{filename}. tz:{df.index.tz} {start} to {end} with {interval} interval")

    return df

def _write_data(df: pd.DataFrame, file_name: str):


    if df.index.tz is None:
        # Index is tz-naive, so localize
        df.index = df.index.tz_localize("UTC")
    else:
        # Index is tz-aware, so convert
        df.index = df.index.tz_convert("UTC")

    file_path = os.path.join(os.path.dirname(__file__), "..", "data", file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path)

    return

def _load_data(file_name: str) -> pd.DataFrame:

    # Check if file exists first
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", file_name)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=["datetime"], index_col="datetime")
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_convert("UTC")
        return df
    else:
        return None

def load_symbol(symbol: str):

    '''
    returns symbol metadata if it exists in data_symbols.json
    else returns None
    '''

    if not symbol in get_symbol_list():  
        return None

    # file path
    json_path = os.path.join(os.path.dirname(__file__), "..", "data_symbols", "data_symbols.json")
    symbols = load_json(json_path)
    return symbols.get(symbol)

def load_json(file_path: str):

    if not os.path.exists(file_path):
        log.error(f"File not found at {file_path}")
        return None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data
    except Exception as e:
        log.error(f"Error decoding JSON from file at {file_path}: {e}")
        return None

def add_symbol_to_json(data):

    json_path = os.path.join(os.path.dirname(__file__), "..", "data_symbols", "data_symbols.json")
    symbols = load_json(json_path)

    return

def get_symbol_list():
    json_path = os.path.join(os.path.dirname(__file__), "..", "data_symbols", "data_symbols.json")
    symbols = load_json(json_path)
    return list(symbols.keys())


if __name__ == "__main__":


    ticker = load_symbol("MNQ=F")
    #df = load_yfinance("MNQ=F", start="2025-08-16", end="2025-09-16", interval="5m")
    df = load_binance("BTC/USDT", start="2023-01-01", end="2023-02-01", interval="5m")

    
    
    #print(load_symbol("MNQ=F"))
    
    
