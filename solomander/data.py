import yfinance as yf
import pandas as pd
import os


try: 
    from .logger import log, stamp, pront
except ImportError: 
    #for running as main script
    from logger import log, stamp, pront

def load_yfinance(ticker: str, start :str = "2023-01-01", end: str= "2023-12-31", interval: str ="1d") -> pd.DataFrame:
 
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
    # define file destination
    dataDir = os.path.join(os.path.dirname(__file__), "..", "data")
    fileName = f"data_{ticker}_{start}_{end}_{interval}.csv".replace("-", "")
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
    df = yf.download(ticker, start=start, end=end, interval=interval)

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
        log.debug(f"Failed to load data for {ticker} from {start} to {end} with interval {interval}.")
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

        log.debug(f"Successfully downloaded {ticker} from {df.index.tz} {start} to {end} of interval {interval}")
        log.debug(f"Saving to file to {filePath}")
        # Save to CSV
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        df.to_csv(filePath)
        return df

   


if __name__ == "__main__":

    df = load_yfinance("MNQ=F", start="2025-08-16", end="2025-09-16", interval="5m")

    pront.info(df.head())