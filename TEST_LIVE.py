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
TIME_FRAME = mt5.TIMEFRAME_M15
CANDLE_BUFFER = 500
POLL_TIME = 0.5  # seconds
TEST_MODE = False



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

    def __init__(self, inputs = None, **kwargs):

        # New variables here
        self.var_atr_period          = int(inputs.get("var_atr_period", 14))
        self.var_atr_mult            = inputs.get("var_atr_mult", 1.2)
        self.var_rr                  = inputs.get("var_rr", 2.0)
        self.var_size                = inputs.get("var_size", 1.0)
        self.var_allowed_open_trades = inputs.get("var_allowed_open_trades", 1)
        self.var_ema_fast            = inputs.get("var_ema_fast",3)
        self.var_sma_slow            = inputs.get("var_sma_slow",300)
        self.var_allow_shorts        = inputs.get("var_allow_shorts", True)
        self.var_allow_long        = inputs.get("var_allow_long", True)
        self.var_threshold_max          = inputs.get("var_threshold_max",1200)
        self.var_threshold_min          = inputs.get("var_threshold_min",400)
   
        # modified variables after init
        super().__init__(**kwargs)
        self._last_trade_day = None  # stores date() of last trade
        self.setting_plot_rows = 5
        

    # ===== indicators / state =====
    def update_data(self):

        self.d_atr = f"d_atr_{self.var_atr_period}"
        self.d_ema_fast = f"d_ema_{self.var_ema_fast}"
        self.d_sma_slow = f"d_ema_{self.var_sma_slow}"
        self.d_vwap = "d_vwap"
        self.d_in_ny = "ny"
        self.d_in_asia = "asia"
        self.d_in_london = "london"
        self.d_crosshighs = "d_crosshighs"
        self.d_gradient_crossup = "d_gradient_crossup"
        self.d_gradient_crossdown = "d_gradient_crossdown"
        self.d_gradient_ema = "d_gradient_ema"
        self.d_vwap_ema_cumsum = "d_vwap_ema_cumsum"
        self.d_sma_gradient = "d_sma_gradient"
        self.d_bull_threshold = "d_bull_thres"
        self.d_bear_threshold = "d_bear_thres"
       


        self.df[self.d_atr] = ta.ATR(self.df, timeperiod = self.var_atr_period)
        self.df[self.d_ema_fast] = ta.SMA(self.df, timeperiod = self.var_ema_fast)
        self.df[self.d_sma_slow] = ta.SMA(self.df, timeperiod = self.var_sma_slow)
        self.df[self.d_vwap] = s.vwap(self.df, mode = "daily")

        sessions = s.sessions(self.df)
        self.df[self.d_in_ny] = sessions['NY']
        self.df[self.d_in_asia] = sessions['TKY']
        self.df[self.d_in_london] = s.timeblock(df, start="7:30", end="13:30", tz="UTC")

        self.df["day"] = self.df.index.date
        self.df[self.d_vwap_ema_cumsum] = s.area_between(self.df[self.d_ema_fast], self.df[self.d_vwap])
        self.df[self.d_gradient_ema] = np.gradient(self.df[self.d_ema_fast])
        #self.df[self.ema_gradient] = self.df[self.ema_gradient].rolling(3, min_periods=1).mean()

        self.df[self.d_gradient_crossup] = pta.cross(self.df[self.d_gradient_ema], pd.Series(0, index = self.df.index))
        self.df[self.d_gradient_crossdown] = pta.cross(pd.Series(0, index = self.df.index), self.df[self.d_gradient_ema])

        self.df[self.d_sma_gradient] = np.gradient(self.df[self.d_sma_slow])

        angle = np.clip(np.degrees(np.arctan(self.df[self.d_sma_gradient])), -60,60)

        self.df[self.d_bull_threshold] = np.interp(angle, [0,60], [self.var_threshold_max, self.var_threshold_min])
        self.df[self.d_bear_threshold] = -self.df[self.d_bull_threshold]



     # ===== plots =====
    def plots(self):
        
        # ==== graph 1 ===
        s.plot_timeblock(self.df, self.d_in_ny, ax = self.axs[0], color="#e9905462", title=self.d_in_ny )
        s.plot_timeblock(self.df, self.d_in_london,ax = self.axs[0], color="#54e96d62", title=self.d_in_london )
        s.plot_timeblock(self.df, self.d_in_asia, ax = self.axs[0], color="#d854e962", title=self.d_in_asia)
        
        a =fplt.plot(self.df[self.d_vwap], ax=self.axs[0], color="#0004ff", legend=self.d_vwap, style = '--', width=2)
        b = fplt.plot(self.df[self.d_ema_fast], ax=self.axs[0], color="#ff0800", legend=self.d_ema_fast, width=1)
        fplt.plot(self.df[self.d_sma_slow], ax=self.axs[0], color="#00ffff", legend=self.d_ema_fast, width=10)
        fplt.fill_between(a,b,color="#ff080039")

        # === graph 2 ====

        fplt.plot(self.df[self.d_vwap_ema_cumsum], ax=self.axs[1], color="#65d40a", legend=self.d_vwap_ema_cumsum, width=2)
        fplt.plot(self.df[self.d_bull_threshold], ax=self.axs[1], color="#65d40a", legend=self.d_bull_threshold, width=1, style = '--')
        fplt.plot(self.df[self.d_bear_threshold], ax=self.axs[1], color="#d40a0a", legend=self.d_bull_threshold, width=1, style = '--')
      
        fplt.plot(pd.Series(0,index = self.df.index), ax=self.axs[1], color="#000000", legend="", style = '--', width=1)
        fplt.plot(pd.Series(self.var_threshold_max,index = self.df.index), ax=self.axs[1], color="#000000", legend="", style = '--', width=1)
        fplt.plot(pd.Series(-self.var_threshold_max,index = self.df.index), ax=self.axs[1], color="#000000", legend="", style = '--', width=1)
       
        
        # gradient cross
        fplt.plot(self.df[self.d_gradient_ema], ax=self.axs[2], color="#1e53c5", legend=self.d_gradient_ema, width=1)
        s.plot_signal(self.df[self.d_gradient_crossup], pd.Series(0,self.df.index), ax = self.axs[2], color =  "#00F80C", style="t", width = 1)
        s.plot_signal(self.df[self.d_gradient_crossdown], pd.Series(0,self.df.index), ax = self.axs[2], color =  "#C71010DC", style="o", width = 1)

        fplt.plot(pd.Series(0,index = self.df.index), ax=self.axs[2], color="#000000", legend="ema_vwap", style = '--', width=1)
        
        # === graph 3 ===
        fplt.plot(self.df[self.d_atr], ax=self.axs[3], color="#e74c3c", legend=self.d_atr)

    # ===== BUY =====

    def buy_condition(self, i):
       
        #return self.data[self.cross_high][i-1] and self.data[self.in_london][i-1] > 0 and self.OPEN_TRADES < self.ALLOWED_OPEN_TRADES
        #return self.data[self.in_ny][i-2] > 0 and self.data[self.in_ny][i-3] < 0
            
        if self.var_allow_long:
            return self.data[self.d_gradient_crossup][i-1] and self.data[self.d_vwap_ema_cumsum][i] > self.data[self.d_bull_threshold][i] and self.OPEN_TRADES < self.var_allowed_open_trades
        
        else:
            return 0
        

    def buy_action(self, i):
        self._last_trade_day = pd.Timestamp(self.data['datetime'][i]).date()
        sl = float(self.data[self.d_atr][i]) * float(self.var_atr_mult)
        tp = sl * float(self.var_rr)
        return self.buy_bracket(i, self.var_size, sl_pips=sl, tp_pips=tp, comments="ORB_L")

    # ===== SELL =====
    def sell_condition(self, i):
        
         if self.var_allow_shorts:
            return self.data[self.d_gradient_crossdown][i-1] and self.data[self.d_vwap_ema_cumsum][i] < self.data[self.d_bear_threshold][i] and self.OPEN_TRADES < self.var_allowed_open_trades
         else:
            return 0

    def sell_action(self, i):
        self._last_trade_day = pd.Timestamp(self.data['datetime'][i]).date()
        sl = float(self.data[self.d_atr][i]) * float(self.var_atr_mult)
        tp = sl * float(self.var_rr)
        return self.sell_bracket(i, self.var_size, sl_pips=sl, tp_pips=tp, comments="ORB_S")

   

params = {  'var_threshold_min': 525, 
            'var_threshold_max': 878,
            'var_atr_mult': 1.03,
            'var_atr_period': 14, 
            'var_rr': 1.22, 
            'var_size': 1.0,
            'var_allowed_open_trades': 1,
            'var_ema_fast': 3,
            'var_sma_slow': 300, 
            'var_allow_shorts': True, 
            'var_allow_long': True}



strategy = OpeningRangeBreakout(inputs = params)

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
    bot = s.DiscordBot(mt5_bot)
    bot.am_ready.wait()  #wait till bot is ready
   
    bot.post_message(f"Hello Jordan 😊 Solomander here!")
    bot.post_message(f"Strategy is now live, using the following settings:")
    bot.post_embed(strategy.discord_settings())
    bot.post_embed(strategy.discord_market())
    bot.post_embed(strategy.discord_inputs())
else:
    matplotlib.use("TkAgg")
    
    
mt5_bot.mt5_stream()











    




