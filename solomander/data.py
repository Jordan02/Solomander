import yfinance as yf
import pandas as pd
import os
import ccxt
import time
import json
from zoneinfo import ZoneInfo
from datetime import datetime

try: 
    from .logger import log, stamp, pront
    from .utils import timedelta_to_str
except ImportError: 
    #for running as main script
    from logger import log, stamp, pront
    from utils import timedelta_to_str


def yfin_load_data(symbol: str, start :str = "2023-11-01", end: str= "2023-12-31", interval: str ="1d", read=True, write=False) -> pd.DataFrame:
 
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
    # ----- Read data -----
    filename = f"yfin_{symbol}_{start}_{end}_{interval}.csv".replace("-", "").replace("/", "-")
    
    if read:
        df = _load_data(filename) 
        if df is not None:
            stamp.success(f"✅ {symbol} loaded from data folder. Opening now queen.")
            return df
        else:
            stamp.warning(f"🚧 {symbol} data cannot be read from data folder, fetching now...")


    # ----- load data -----
    try:
        df = yf.download(symbol, start=start, end=end, interval=interval)
        symbol_info = yf.Ticker(symbol).info

    except Exception as e:
        log.error(f"❌ Error downloading from yfinance: {e}")
        return pd.DataFrame()
    
    # ----- Update symbol data -----
    data_tz = symbol_info.get("exchangeTimezoneName")
    data_tz_utc = tz_to_utc_offset(data_tz)
    #print(data_tz) # data/timezone
    # print(df.index.tz) # data has no timezone

    json_path = os.path.join(os.path.dirname(__file__), "..", "data_symbols", "data_symbols.json")
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
    else:
        all_data = {}

    # Update or create symbol entry
    if symbol not in all_data:
        all_data[symbol] = {}

    all_data[symbol]["symbol"] = symbol
    all_data[symbol]["data_tz"] = data_tz
    all_data[symbol]["data_tz_utc"] = data_tz_utc
    all_data[symbol]["server"] = symbol_info["exchange"]
    stamp.critical("You're Using yfinance for data - you NEED to MANAULLY update symbol info json")

    # Save it back
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)


    # ----- format data -----

    df.index = pd.to_datetime(df.index) # convert index to datetime
    # remove second column name if it is a MultiIndex, and lowercase all column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        df.columns = df.columns.str.lower()
    else:
        df.columns = df.columns.str.lower()

    df = df.reindex(columns=["open", "close", "high", "low", "volume"])
    df.columns.name = None
    df.reset_index(drop=True)
    df.index.name = "datetime"

    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize(data_tz)
    else:
        df.index = df.index.tz_convert(data_tz)
        
    # ----- write data -----

    if write:
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
    if read_symbol(symbol) is None:
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



# ===== data loading ===== 

def _write_data(df: pd.DataFrame, file_name: str):

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
        return df
    else:
        return None


# ===== symbol loading =====

def read_symbol(symbol: str):

    '''
    returns symbol metadata if it exists in data_symbols.json
    else returns None
    '''
    # file path
    json_path = os.path.join(os.path.dirname(__file__), "..", "data_symbols", "data_symbols.json")

    try:
        with open(json_path, "r") as f:
            all_data = json.load(f)
            symbol_info = all_data.get(symbol)
            return symbol_info
    except Exception as e:
        log.error(f"Error decoding JSON from file at {json_path}: {e}")
        return None

def write_symbol(symbol_data):

    '''
    returns symbol metadata if it exists in data_symbols.json
    else returns None
    '''
    # file directory
    json_path = os.path.join(os.path.dirname(__file__), "..", "data_symbols", "data_symbols.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # make sure file has contents, open to read
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
    else:
        all_data = {}

    # add or update data
    symbol_key = symbol_data.get("symbol", "unkown")
    all_data[symbol_key] = symbol_data

    # overwrite the whole file with updated dict
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
            return True
    except Exception as e:
        log.error(f"Error writting symbol data: {e}")
        return False

# ===== other ======

def tz_from_utx_offset(offset_hours: int | float) -> str:

    """
    Convert a numeric UTC offset (e.g. -3, +2) into a named timezone string.
    Falls back to an Etc/GMT zone if unknown.
    """
    wrapped_hours = (offset_hours - 1) % 12 + 1

    offset_map = {
        -12: "Etc/GMT+12",
        -11: "Pacific/Niue",
        -10: "Pacific/Honolulu",
        -9:  "America/Anchorage",
        -8:  "America/Los_Angeles",
        -7:  "America/Denver",
        -6:  "America/Chicago",
        -5:  "America/New_York",
        -4:  "America/Halifax",
        -3:  "America/Argentina/Buenos_Aires",
        -2:  "America/Noronha",
        -1:  "Atlantic/Azores",
         0:  "UTC",
         1:  "Europe/Lisbon",           # or Europe/London in winter
         2:  "Europe/Prague",           # FTMO, IC Markets, etc.
         3:  "Europe/Moscow",           # UTC+3, many brokers use this
         4:  "Asia/Dubai",
         5:  "Asia/Karachi",
         6:  "Asia/Dhaka",
         7:  "Asia/Bangkok",
         8:  "Asia/Singapore",          # or Asia/Hong_Kong
         9:  "Asia/Tokyo",
         10: "Australia/Sydney",
         11: "Pacific/Noumea",
         12: "Pacific/Auckland"
    }

    # round offset in case of small decimals like 2.0 or -3.5
    offset_int = int(round(wrapped_hours))
    tz_name = offset_map.get(offset_int, "unkown")

    sign = "+" if offset_int >= 0 else ""
    utc_name = f"UTC{sign}{offset_int}"

    return tz_name, utc_name

def tz_to_utc_offset(tz_name: str) -> str:
    """Convert timezone name to UTC±X offset string"""
    try:
        tz = ZoneInfo(tz_name)
        now = datetime.now(tz)
        offset_hours = now.utcoffset().total_seconds() / 3600
        sign = "+" if offset_hours >= 0 else "-"
        return f"UTC{sign}{abs(offset_hours):.0f}"
    except Exception as e:
        print(f"⚠️ Could not resolve timezone '{tz_name}': {e}")
        return "UTC+0"




if __name__ == "__main__":


    df = yfin_load_data("MNQ=F")
    
    
    #print(load_symbol("MNQ=F"))
    
    
