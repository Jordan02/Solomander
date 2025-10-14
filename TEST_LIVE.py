import solomander as s
from solomander.logger import log, stamp, pront
from solomander.baseStrategy import Strategy
from solomander.backtester import Backtester
from matplotlib import pyplot as plt
from scipy.stats import skewnorm, norm
import MetaTrader5 as mt5
from dotenv import load_dotenv
import os

import matplotlib


import pandas as pd
import finplot as fplt
import talib
import talib.abstract as ta
import pandas_ta as pta
import numpy as np
import optuna
import optuna.visualization as vis
import optuna.visualization.matplotlib as vism



DISCORD_BOT = True
SYMBOL = "US100.cash"
TIME_FRAME = mt5.TIMEFRAME_M1
CANDLE_BUFFER = 500
POLL_TIME = 0.5  # seconds
TEST_MODE = True



pd.set_option("display.max_columns", None)

s.mt5_login()
ticker = s.mt5_symbol_info(SYMBOL)
df = s.mt5_hdata(SYMBOL, TIME_FRAME, candle_lookback=2000)



class strat1(Strategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.SMA_FAST = kwargs.get("SMA_FAST", 10)
        self.SMA_SLOW = kwargs.get("SMA_SLOW", 50)
        self.ATR_MULTIPLIER = kwargs.get("ATR_MULTIPLIER", 1.5)
        self.SIZE = kwargs.get("SIZE", 1.0)
        self.ALLOWED_OPEN_TRADES = kwargs.get("ALLOWED_OPEN_TRADES", 1)
        self.RR = kwargs.get("RR", 2.0)
        pass

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
        return self.data['crossover'][i-1] > 0 and self.OPEN_TRADES < self.ALLOWED_OPEN_TRADES

    def buy_action(self, i):
        
        sl = self.data['atr'][i]*self.ATR_MULTIPLIER
        tp = sl * self.RR
        
        return self.buy_bracket(i, self.SIZE, sl_pips=sl, tp_pips=tp, comments='B')
    
    # ===== SELL LOGIC =====
    def sell_condition(self, i):

        time_cond = self.data['NY'][i] > 0 # in ny session  
        return self.data['crossunder'][i-1] > 0 and self.OPEN_TRADES < self.ALLOWED_OPEN_TRADES
    
    def sell_action(self, i):
    
        sl = self.data['atr'][i]*self.ATR_MULTIPLIER
        tp = sl * self.RR
        
        return self.sell_bracket(i, self.SIZE, sl_pips=sl, tp_pips=tp, comments='S')

    # ===== Visualization =====

    def plots(self):
        super().plots(rows=3)

        fplt.plot(self.df['SMA_slow'] , ax=self.axs[0], color="#ff6a00", legend=f"SMA {self.SMA_SLOW}")
        fplt.plot(self.df['SMA_fast'] , ax=self.axs[0], color="#00ff6a", legend=f"SMA {self.SMA_FAST}")
        fplt.plot(self.df['atr'], ax=self.axs[1], color="#00ff6a", legend="ATR")

        s.plot_timeband(self.df, 'NY', ax=self.axs[0], color="#a8a8a83d", title="NY")


best_params={"atr_multiplier": 2, "sma_fast": 17, "sma_slow": 54, "rr": 1.3, "open_trades": 4}

strategy = strat1(
                    ATR_MULTIPLIER= best_params['atr_multiplier'],
                    RR=best_params['rr'],
                    SMA_FAST= best_params['sma_fast'],
                    SMA_SLOW=best_params['sma_slow'],
                    ALLOWED_OPEN_TRADES= best_params['open_trades'],
                    SIZE=1)


backtest_strategy = Backtester(strategy,df,ticker)
results = backtest_strategy.execute()


mt5_bot = s.MT5_live(results, 
                     SYMBOL, 
                     TIME_FRAME, 
                     CANDLE_BUFFER, 
                     POLL_TIME, 
                     test_mode=TEST_MODE
                     )

if DISCORD_BOT:
    matplotlib.use("Agg")  
    bot = s.DiscordBot(mt5_bot)
    bot.am_ready.wait()  #wait till bot is ready
else:
    matplotlib.use("TkAgg")

mt5_bot.mt5_stream()






    




