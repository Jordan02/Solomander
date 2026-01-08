from solomander.baseStrategy import Strategy
import solomander as s
import pandas_ta as pta
import finplot as fplt

from solomander.backtester import Backtester

import MetaTrader5 as mt5

class sma_crossover_strategy(Strategy):

    # simple sma cross over
    def __init__(self, inputs = {}, **kwargs):

        # New variables here
        self.var_rr                  = inputs.get("var_rr", 2.0)                 # risk to reward
        self.var_size                = inputs.get("var_size", 1.0)               # trade size
        self.var_allowed_open_trades = inputs.get("var_allowed_open_trades", 1)  # max open trades
        self.var_sma_fast            = inputs.get("var_ema_fast",6)              # ema fast
        self.var_sma_slow            = inputs.get("var_sma_slow",60)             # sma slow
   
        # modified variables after init
        super().__init__(**kwargs)
        self.setting_plot_rows = 3

    def update_data(self):
        
         # indicator column names
        self.d_sma_fast = f"sma_fast_{self.var_sma_fast}"
        self.d_sma_slow = f"sma_slow_{self.var_sma_slow}"
        self.d_crossup = f"cross_up"
        self.d_crossdown = f"cross_down"
        self.d_atr= f"d_atr_14"

         # variables
        self.df[self.d_sma_fast] = pta.sma(self.df["close"], self.var_sma_fast)
        self.df[self.d_sma_slow] = pta.sma(self.df["close"], self.var_sma_slow)
        self.df[self.d_atr] = pta.atr(self.df["high"], self.df["low"], self.df["close"], timeperiod=14)
        self.df[self.d_crossup ] = pta.cross(self.df[self.d_sma_fast], self.df[self.d_sma_slow])
        self.df[self.d_crossdown ] = pta.cross(self.df[self.d_sma_slow], self.df[self.d_sma_fast])

    def plots(self):
         
         # graph 1 = candlestick chart
        fplt.plot(self.df[self.d_sma_fast], ax=self.axs[0], color="#65d40a", legend=self.d_sma_fast, width=1, style = '-')
        fplt.plot(self.df[self.d_sma_slow], ax=self.axs[0], color="#d41b0a", legend=self.d_sma_slow, width=1, style = '-')

         # graph 2 = added signal chart
        fplt.plot(self.df[self.d_crossup], ax=self.axs[1], color="#65d40a", legend=self.d_crossup, width=1, style = '-')
        fplt.plot(self.df[self.d_crossdown], ax=self.axs[1], color="#d41b0a", legend=self.d_crossdown, width=1, style = '-')

         

    def buy_condition(self, i):
        return self.data[self.d_crossup][i-1] > 0 and self.OPEN_TRADES < self.var_allowed_open_trades
    
    def sell_condition(self, i):
        return self.data[self.d_crossdown][i-1] > 0 and self.OPEN_TRADES < self.var_allowed_open_trades
    
    def buy_action(self, i):
        sl = float(self.data[self.d_atr][i-1])
        tp = sl * float(self.var_rr)
        return self.buy_bracket(i, self.var_size, sl_pips=sl, tp_pips=tp)
    
    def sell_action(self, i):
        sl = float(self.data[self.d_atr][i-1])
        tp = sl * float(self.var_rr)
        return self.sell_bracket(i, self.var_size, sl_pips=sl, tp_pips=tp)
        

# Data

# Market Settings
SYMBOL = "US100.cash"
TIME_FRAME = mt5.TIMEFRAME_M15  # M5, M15, H1
CANDLE_LOOKBACK = 20000
CANDLE_OFFSET = 0
s.mt5_login()
ticker = s.mt5_load_symbol(SYMBOL)
df = s.mt5_hdata(SYMBOL, TIME_FRAME, candle_lookback=CANDLE_LOOKBACK, candle_offset=CANDLE_OFFSET, refresh_data = True)


## RUN strategy

strategy = sma_crossover_strategy()
study = Backtester(strategy, df, ticker)
result = study.execute()

study.show()

