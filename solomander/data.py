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
    # check there is valid symbol data available
    if load_symbol(symbol) is None:
        log.error(f"No symbol data available for: {symbol}, make sure it is in data/symbol_data.json")
        return pd.DataFrame()

    # define file destination
    dataDir = os.path.join(os.path.dirname(__file__), "..", "data")
    fileName = f"data_{symbol}_{start}_{end}_{interval}.csv".replace("-", "")
    filePath = os.path.join(dataDir, fileName)

    # Check if file exists first
    if os.path.exists(filePath):
        df = pd.read_csv(filePath, parse_dates=["datetime"], index_col="datetime")
        df.index = pd.to_datetime(df.index)

        if df.index.tz is None:
            # Index is tz-naive, so localize
            df.index = df.index.tz_localize("UTC")
        else:
            # Index is tz-aware, so convert
            df.index = df.index.tz_convert("UTC")

        log.debug(f"File found in data/{fileName} Opening now queen.")
        log.debug(f"Data from {df.index.tz} {df.index[0]} to {df.index[-1]} of interval {interval}")
        return df

    log.debug(f"Downloading data")
    df = yf.download(symbol, start=start, end=end, interval=interval)

    # === finplot requires date (or datetime) to be the index ===
    df.index = pd.to_datetime(df.index)

    # remove second column name if it is a MultiIndex, and lowercase all column names
    # === TA lib needs column names to be lowercase ===
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        df.columns = df.columns.str.lower()
    else:
        df.columns = df.columns.str.lower()

    # === Finplot requires OCHL order ===
    df = df.reindex(columns=["open", "close", "high", "low", "volume"])

    # rename index
    df.columns.name = None
    df.reset_index(drop=True)
    df.index.name = "datetime"


    if df is None:
        log.debug(f"Failed to load data for {symbol} from {start} to {end} with interval {interval}.")
        df = pd.DataFrame()  # Return empty DataFrame if no data
        df.index = pd.to_datetime(df.index)
        return df
    
    else:
        df.index = pd.to_datetime(df.index)

        if df.index.tz is None:
            # Index is tz-naive, so localize
            df.index = df.index.tz_localize("UTC")
        else:
            # Index is tz-aware, so convert
            df.index = df.index.tz_convert("UTC")

        log.debug(f"Successfully downloaded {symbol} from {df.index.tz} {start} to {end} of interval {interval}")
        log.debug(f"Saving to file to {filePath}")
        # Save to CSV
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        df.to_csv(filePath)
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
    # mapping for Binance intervals
    interval_map = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
        "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M"
    }
    if interval not in interval_map:
        raise ValueError(f"Unsupported interval {interval}")

    # define file destination
    dataDir = os.path.join(os.path.dirname(__file__), "..", "data")
    fileName = f"data_{symbol.replace('/', '')}_{start}_{end}_{interval}.csv".replace("-", "")
    filePath = os.path.join(dataDir, fileName)

    # if cached file exists
    if os.path.exists(filePath):
        df = pd.read_csv(filePath, parse_dates=["datetime"], index_col="datetime")
        df.index = pd.to_datetime(df.index).tz_convert("UTC")
        
        return df

    log.debug(f"Downloading {symbol} data from Binance")

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
        log.error(f"No data returned for {symbol}")
        return pd.DataFrame()

    # convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
    df = df[~df.index.duplicated(keep="last")] # REMOVE DUPLICATES
    df.set_index("datetime", inplace=True)

    # reindex to OCHL order (your convention)
    df = df.reindex(columns=["open", "close", "high", "low", "volume"])

    # save
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    df.to_csv(filePath)

    log.debug(f"Saved {symbol} {interval} data to {filePath} ({df.index[0]} → {df.index[-1]})")

    return df


def load_symbol(symbol: str):

    # file path
    json_path = os.path.join(os.path.dirname(__file__), "..", "data", "symbol_data.json")
    if not os.path.exists(json_path):
        log.error(f"Symbols file not found at {json_path}")
        return None
    
    # laod json
    try:
        with open(json_path, "r") as f:
            symbols = json.load(f)
            if symbol not in symbols:
                log.error(f"Symbol {symbol} not found in symbols file.")
                return None
    except Exception as e:
        log.error(f"Error decoding JSON from symbols file at {json_path}: {e}")
        return None
    
    return symbols.get(symbol)


if __name__ == "__main__":


    ticker = load_symbol("MNQ=F")
    df = load_yfinance(ticker['symbol'], start="2025-08-16", end="2025-09-16", interval="5m")
    #df = load_binance("BTC/USDT", start="2023-01-01", end="2023-02-01", interval="5m")

    time_step = df.index.to_series().diff().dropna().min()

    print(ticker)
    print(f"Minimum time step: {time_step}")
    print(f"Minimum time step: {timedelta_to_str(time_step)}")
