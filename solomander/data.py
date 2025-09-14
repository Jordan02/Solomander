import yfinance as yf
import pandas as pd
from solomander.logger import log, stamp, pront


def test():
    print("Test function in data module")

def load_data(ticker: str, start :str = "2023-01-01", end: str= "2023-12-31", interval: str ="1d") -> pd.DataFrame:
 
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

    df = yf.download(ticker, start=start, end=end, interval=interval)
    log.debug(f"Data for {ticker} from {start} to {end} with interval {interval} loaded.")

    if df is None:
        df = pd.DataFrame()  # Return empty DataFrame if no data
        return df
    

    return df