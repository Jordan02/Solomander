import finplot as fplt
import talib
import talib.abstract as ta
import pandas_ta as pta
import pandas as pd

from typing import final

try:
    from .logger import log, stamp, pront
    from .data import load_yfinance
except ImportError:
    from logger import log, stamp, pront
    from data import load_yfinance
    
    
class Strategy:
    def __init__(self, df : pd.DataFrame):
        self.df = df
        
    def buy(self):
        print("Buy logic here")
        return

    @final
    def run(self):

        for idx, row in self.df.iterrows():
            print(idx, row['SMA_slow'], row['SMA_fast'])

        return
    
if __name__ == "__main__":
    pront.info("Solomander base strategy running.")

    
    df = load_yfinance("MNQ=F", start="2025-08-16", end="2025-09-16", interval="5m")

    SMA_SLOW = 50
    SMA_FAST = 20
    df['SMA_slow'] = ta.SMA(df,timeperiod=SMA_SLOW)
    df['SMA_fast'] = ta.SMA(df,timeperiod=SMA_FAST)

