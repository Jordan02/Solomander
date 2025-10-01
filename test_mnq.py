import solomander as s
from solomander.logger import log, stamp, pront
from solomander.strategy import Strategy
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

s.mt5_login()
ticker = s.mt5_symbol_info(SYMBOL)
df = s.mt5_hdata(SYMBOL, mt5.TIMEFRAME_M5, lookback=4000)

#df = s.load_yfinance(SYMBOL, start="2025-08-16", end="2025-09-16", interval="5m")
#ticker = s.load_symbol(SYMBOL)


print(df.head(10))


# ==== DATA AND INDICATORS =====


df['NY'] = s.sessions(df)['NY']
df['atr'] = ta.ATR(df, timeperiod=14)


# ==== STRATEGY EXECUTION =====

class strat1(Strategy):

    # add Input and settings here, so they can be intellisensed
    FEE: float
    LEVERAGE: float
    ATR_MULTIPLIER: float
    RR: float
    SMA_FAST: int
    SMA_SLOW: int
    SIZE: float

    # ===== SET STRATEGY PARAMETERS =====
    def add_market_data(self):

        # update only those that are dynamic during optimsation, fix others outside
        self.df['SMA_slow'] = ta.SMA(self.df['close'], timeperiod=self.SMA_SLOW)
        self.df['SMA_fast'] = ta.SMA(self.df['close'], timeperiod=self.SMA_FAST)
        self.df['crossover'] = pta.cross(self.df['SMA_fast'], self.df['SMA_slow'])
        self.df['crossunder'] = pta.cross(self.df['SMA_slow'], self.df['SMA_fast'])

        return

        
    # ===== BUY LOGIC =====
    def buy_condition(self, i):
        time_cond = self.data['NY'][i] > 0 # in ny session 
        return self.data['crossover'][i] > 0 and self.OPEN_TRADES < 3

    def buy_action(self, i):
        
        pre_i = max(0, i)
        sl = self.data['atr'][pre_i]*self.ATR_MULTIPLIER
        tp = sl * self.RR
        self.buy_bracket(i, self.SIZE, sl_pips=sl, tp_pips=tp, comments='B')
        return 
    
    # ===== SELL LOGIC =====
    def sell_condition(self, i):

        time_cond = self.data['NY'][i] > 0 # in ny session  
        return self.data['crossunder'][i] > 0 and self.OPEN_TRADES < 3
    
    def sell_action(self, i):
    
        pre_i = max(0, i)
        sl = self.data['atr'][pre_i]*self.ATR_MULTIPLIER
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



# ==== OPTIMIZATION =====


# ==== EXECUTION =====


#guess_1={"atr_multiplier": 1.83, "sma_fast": 6, "sma_slow": 46}
test_params={"atr_multiplier": 1.12, "sma_fast": 8, "sma_slow": 45} # 5-15SR best so far

st = strat1(df,
            ticker,
            ATR_MULTIPLIER= test_params['atr_multiplier'],
            RR=1.5,
            SMA_FAST= test_params['sma_fast'],
            SMA_SLOW= test_params['sma_slow'],
            SIZE=4)


st.execute()
st.show()
st.print_metrics()

#test_params={'SMA_FAST':1,'SMA_SLOW':1}
#test_params={'ATR_MULTIPLIER':0.01, 'RR': 0.02}
#s.monte_carlo(st.tf, runs=300, mode='bootstrap', seed=156, params=st.INPUT_PARAMS)
#s.noise_test(st, test_params=test_params, nudges=3)
print(st.tf)


plt.show()




