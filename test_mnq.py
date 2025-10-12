import solomander as s
from solomander.logger import log, stamp, pront
from solomander.baseStrategy import Strategy
from solomander.backtester import Backtester
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
df = s.mt5_hdata(SYMBOL, mt5.TIMEFRAME_M5, candle_lookback=2000)

#df = s.load_yfinance(SYMBOL, start="2025-08-16", end="2025-09-16", interval="5m")
#ticker = s.load_symbol(SYMBOL)


class strat1(Backtester):

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


# ==== OPTIMIZATION =====

pnl_curve = []
display_values = []

def objective(trial):

    # Define the hyperparameters to optimize
    atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 3.0)
    rr = trial.suggest_float("rr", 1.0, 3.0)
    sma_fast = trial.suggest_int("sma_fast", 6, 30)
    sma_slow = trial.suggest_int("sma_slow", 40, 150)
    open_trades = trial.suggest_int("open_trades", 1, 10)

    # Create and run the strategy
    st = strat1(  df,
                  ticker,
                  ATR_MULTIPLIER=atr_multiplier, 
                  RR=rr, 
                  SMA_FAST=sma_fast, 
                  SMA_SLOW=sma_slow,
                  ALLOWED_OPEN_TRADES=open_trades,
                  SIZE=1 )
    
    st.execute()

    pnl_curve.append(st.tf['pnl'].cumsum().to_numpy())
    display_values.append({  "atr_multiplier": atr_multiplier, 
                             "sma_fast": sma_fast,
                             "sma_slow": sma_slow, 
                             "mdd": st.MAX_DRAWDOWN, 
                             "sr": st.SHARPE_RATIO_ANNUAL,
                             "pnl": st.PNL,
                             "open_trades": open_trades,
                             "rr": rr})

    return st.SHARPE_RATIO_ANNUAL

#guess_1={"atr_multiplier": 1.83, "sma_fast": 6, "sma_slow": 46}
guess_2={"atr_multiplier": 2.67, "sma_fast": 23, "sma_slow": 127, "rr": 2.76, "open_trades": 1}
guess_3={"atr_multiplier": 1.62, "sma_fast": 17, "sma_slow": 55, "rr": 1.5, "open_trades": 3}
guess_4={"atr_multiplier": 2.35, "sma_fast": 25, "sma_slow": 58, "rr": 1.78, "open_trades": 3}
guess_5={"atr_multiplier": 2.07, "sma_fast": 17, "sma_slow": 56, "rr": 2.17, "open_trades": 1}
guess_11={"atr_multiplier": 1.44, "sma_fast": 18, "sma_slow": 108, "rr": 1.68, "open_trades": 8}


if 0:
    study = s.Optimise( objective,
                        n_trials=10,
                        n_jobs=1,
                        guess=[guess_2, guess_3, guess_4, guess_5, guess_11],
                        direction = 'maximize',
                        target="SR")
    best_params = study.execute()
    study.show(pnl_curves=pnl_curve, info=display_values)


test_params={"atr_multiplier": 2.07, "sma_fast": 17, "sma_slow": 56, "rr": 2.17, "open_trades": 1}

# ==== EXECUTION =====


st = strat1(df,
            ticker,
            ATR_MULTIPLIER= test_params['atr_multiplier'],
            RR=test_params['rr'],
            SMA_FAST= test_params['sma_fast'],
            SMA_SLOW= test_params['sma_slow'],
            ALLOWED_OPEN_TRADES=test_params['open_trades'],
            SIZE=3)



st.execute()
st.show()
st.print_metrics()

#test_params={'SMA_FAST':1,'SMA_SLOW':1}
#test_params={'ATR_MULTIPLIER':0.01, 'RR': 0.02}
s.monte_carlo(st.tf, runs=300, mode='bootstrap', seed=156, params=st.INPUT_PARAMS)
#s.noise_test(st, test_params=test_params, nudges=3)
print(st.tf)


plt.show()




