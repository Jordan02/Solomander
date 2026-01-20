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
DISCORD_CHANNEL = 1462789469630365877



pd.set_option("display.max_columns", None)

s.mt5_login()
ticker = s.mt5_load_symbol(SYMBOL)
df = s.mt5_hdata(SYMBOL, TIME_FRAME, candle_lookback=2000)




class OpeningRangeBreakout(Strategy):
    """
    Opening Range Breakout (ORB)
    - Define opening range as the first `OR_PERIOD` NY-session bars each day.
    - Go long when price **crosses above** OR high after the OR is complete.
    - Go short when price **crosses below** OR low after the OR is complete.
    - One trade per day; ATR-based SL/TP.
    """

    def __init__(self, inputs = {}, **kwargs):

        # New variables here
        self.var_atr_period          = int(inputs.get("var_atr_period", 14))
        self.var_atr_mult            = inputs.get("var_atr_mult", 1.2)
        self.var_rr                  = inputs.get("var_rr", 2.0)
        self.var_size                = inputs.get("var_size", 1.0)
        self.var_allowed_open_trades = inputs.get("var_allowed_open_trades", 10)
  
        # modified variables after init
        super().__init__(**kwargs)
        self._last_trade_day = None  # stores date() of last trade
        

    # ===== indicators / state =====
    def update_data(self):

        self.d_atr = f"d_atr_{self.var_atr_period}"
        self.df[self.d_atr] = ta.ATR(self.df, timeperiod = self.var_atr_period)
       

     # ===== plots =====
    def plots(self):
        
      pass
    
    # ===== BUY =====

    def buy_condition(self, i):
       
        "buy every candle - live will handle this too"
            
        return 1
        

    def buy_action(self, i):

        self._last_trade_day = pd.Timestamp(self.data['datetime'][i]).date()
        #sl = float(self.data[self.d_atr][i])
        sl = 10
        #tp = sl * float(self.var_rr)
        tp = 10
        return self.buy_bracket(i, self.var_size, sl_pips=sl, tp_pips=tp, comments="test_buy_all_candle")


strategy = OpeningRangeBreakout()

#backtest_strategy = Backtester(strategy,df,ticker)
#results = backtest_strategy.execute()


mt5_bot = s.MT5_live(strategy, 
                     SYMBOL, 
                     TIME_FRAME, 
                     CANDLE_BUFFER, 
                     POLL_TIME, 
                     test_mode=TEST_MODE
                     )

if DISCORD_BOT:
    matplotlib.use("Agg")  
    bot = s.DiscordBot(mt5_bot, DISCORD_CHANNEL)
    bot.am_ready.wait()  #wait till bot is ready
   
    bot.post_message(f"Hello Jordan 😊 Solomander here!")
    bot.post_message(f"Strategy is now live, using the following settings:")
    bot.post_embed(strategy.discord_settings())
    bot.post_embed(strategy.discord_market())
    bot.post_embed(strategy.discord_inputs())
else:
    matplotlib.use("TkAgg")
    
    
mt5_bot.mt5_stream()











    




