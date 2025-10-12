import solomander as s
from solomander.logger import log, stamp, pront
from solomander.baseStrategy import Strategy
from solomander.backtester import Backtester
from solomander.mt5 import MT5_live
from matplotlib import pyplot as plt
from scipy.stats import skewnorm, norm
import MetaTrader5 as mt5
from dotenv import load_dotenv
import os

import pandas as pd
import finplot as fplt
import talib
import talib.abstract as ta
import pandas_ta as pta
import numpy as np
import optuna
import optuna.visualization as vis
import optuna.visualization.matplotlib as vism

pd.set_option("display.max_columns", None)


SYMBOL = "US100.cash"

class strat1(MT5_live):

    # add Input and settings here, so they can be intellisensed
    FEE: float
    LEVERAGE: float
    ATR_MULTIPLIER: float
    RR: float
    SMA_FAST: int
    SMA_SLOW: int
    SIZE: float
    ALLOWED_OPEN_TRADES: int

    # ===== SET STRATEGY PARAMETERS =====
    def update_data(self):

        # update only those that are dynamic during optimsation, fix others outside
        self.df['SMA_slow'] = ta.SMA(self.df['close'], timeperiod=self.SMA_SLOW)
        self.df['SMA_fast'] = ta.SMA(self.df['close'], timeperiod=self.SMA_FAST)
        self.df['crossover'] = pta.cross(self.df['SMA_fast'], self.df['SMA_slow'])
        self.df['crossunder'] = pta.cross(self.df['SMA_slow'], self.df['SMA_fast'])
        self.df['NY'] = s.sessions(self.df)['NY']
        self.df['atr'] = ta.ATR(self.df, timeperiod=14)

        return

    # ===== BUY LOGIC =====
    def buy_condition(self, i):
        time_cond = self.data['NY'][i] > 0 # in ny session 
        return self.data['crossover'][i-1] > 0 and self.OPEN_TRADES < self.ALLOWED_OPEN_TRADES

    def buy_action(self, i):
        
        sl = self.data['atr'][i]*self.ATR_MULTIPLIER
        tp = sl * self.RR
        self.buy_bracket(i, self.SIZE, sl_pips=sl, tp_pips=tp, comments='B')
        return 
    
    # ===== SELL LOGIC =====
    def sell_condition(self, i):

        time_cond = self.data['NY'][i] > 0 # in ny session  
        return self.data['crossunder'][i-1] > 0 and self.OPEN_TRADES < self.ALLOWED_OPEN_TRADES
    
    def sell_action(self, i):
    
        sl = self.data['atr'][i]*self.ATR_MULTIPLIER
        tp = sl * self.RR
        self.sell_bracket(i, self.SIZE, sl_pips=sl, tp_pips=tp, comments='S')
        return 

    # ===== Visualization =====

    def plots(self):
        super().plots(rows=3)

        fplt.plot(self.df['SMA_slow'] , ax=self.axs[0], color="#ff6a00", legend=f"SMA {self.SMA_SLOW}")
        fplt.plot(self.df['SMA_fast'] , ax=self.axs[0], color="#00ff6a", legend=f"SMA {self.SMA_FAST}")
        fplt.plot(self.df['atr'], ax=self.axs[1], color="#00ff6a", legend="ATR")

        s.plot_timeband(self.df, 'NY', ax=self.axs[0], color="#a8a8a83d", title="NY")


# ==== EXECUTION =====

params={"atr_multiplier": 2.07, "sma_fast": 17, "sma_slow": 56, "rr": 2.17, "open_trades": 1}

st = strat1("US100.cash",
            timeframe=mt5.TIMEFRAME_M5,
            candle_buffer=500,
            poll_interval=0.5,
            test_mode=True,
            ATR_MULTIPLIER= params['atr_multiplier'],
            RR=params['rr'],
            SMA_FAST= params['sma_fast'],
            SMA_SLOW= params['sma_slow'],
            ALLOWED_OPEN_TRADES=params['open_trades'],
            SIZE=3
            )



bot = s.DiscordBot(st)
st.mt5_stream()







