import solomander as s
from solomander.logger import log, stamp, pront
from solomander.strategy import Strategy
from matplotlib import pyplot as plt
from scipy.stats import skewnorm, norm

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


# ==== DATA AND INDICATORS =====
df = s.load_yfinance("MNQ=F", start="2025-08-16", end="2025-09-16", interval="5m")
df['NY'] = s.sessions(df)['NY']
df['vwap'] = s.vwap(df, mode="daily")
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
    def __init__(self, df : pd.DataFrame, **kwargs):
        super().__init__(df, **kwargs)

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
        self.bracket_order(i,'buy', self.SIZE, sl_pips=sl, tp_pips=tp, comments='B')
        return 
    
    # ===== SELL LOGIC =====
    def sell_condition(self, i):

        time_cond = self.data['NY'][i] > 0 # in ny session  
        return self.data['crossunder'][i] > 0 and self.OPEN_TRADES < 3
    
    def sell_action(self, i):
    
        pre_i = max(0, i)
        sl = self.data['atr'][pre_i]*self.ATR_MULTIPLIER
        tp = sl * self.RR
        self.bracket_order(i,'sell', self.SIZE, sl_pips=sl, tp_pips=tp, comments='S')
        return 

    # ===== Visualization =====

    def plots(self):
        super().plots(rows=3)

        fplt.plot(self.df['SMA_slow'] , ax=self.axs[0], color="#ff6a00", legend=f"SMA {self.SMA_SLOW}")
        fplt.plot(self.df['SMA_fast'] , ax=self.axs[0], color="#00ff6a", legend=f"SMA {self.SMA_FAST}")
        fplt.plot(self.df['vwap'], ax=self.axs[0], color="#219bec", legend="VWAP")
        fplt.plot(self.df['atr'], ax=self.axs[1], color="#00ff6a", legend="ATR")

        s.plot_timeband(self.df, 'NY', ax=self.axs[0], color="#a8a8a83d", title="NY")


# ==== OPTIMISATION =====

pnl_curve = []
display_values = []

def objective(trial):

    # Define the hyperparameters to optimize
    atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 3.0)
    sma_fast = trial.suggest_int("sma_fast", 6, 30)
    sma_slow = trial.suggest_int("sma_slow", 40, 60)

    # Create and run the strategy
    st = strat1(  df, 
                  FEE = 1.74, 
                  LEVERAGE = 2,
                  ATR_MULTIPLIER=atr_multiplier, 
                  RR=1.5, 
                  SMA_FAST=sma_fast, 
                  SMA_SLOW=sma_slow,
                  SIZE=1 )
    
    st.execute()

    pnl_curve.append(st.tf['pnl'].cumsum().to_numpy())
    display_values.append({  "atr_multiplier": atr_multiplier, 
                             "sma_fast": sma_fast,
                             "sma_slow": sma_slow, 
                             "mdd": st.MAX_DRAWDOWN, 
                             "sr": st.SHARPE_RATIO_ANNUAL,
                             "pnl": st.PNL})

    return st.SHARPE_RATIO_ANNUAL

#guess_1={"atr_multiplier": 1.83, "sma_fast": 6, "sma_slow": 46}
guess_1={"atr_multiplier": 2.17, "sma_fast": 11.0, "sma_slow": 55}
guess_2={"atr_multiplier": 4, "sma_fast": 7, "sma_slow": 56}


study = s.Optimise( objective,
                    n_trials=200, 
                    n_jobs=1, 
                    guess=[], 
                    direction = 'maximize',
                    target="Sharpe Ratio")
best_params = study.execute()
study.show(pnl_curves=pnl_curve, info=display_values)


st = strat1(df,
            FEE=1.74, 
            LEVERAGE = 2,
            ATR_MULTIPLIER= best_params['atr_multiplier'],
            RR=1.5,
            SMA_FAST= best_params['sma_fast'],
            SMA_SLOW= best_params['sma_slow'],
            SIZE=1)

# ATR_MULTIPLIER=1.83 SMA_FAST=6 SMA_SLOW=46

st.execute()
#st.show()
st.print_metrics()

s.monte_carlo(st.tf, runs=500, mode='bootstrap', seed=156, params=st.INPUT_PARAMS)
plt.show()




