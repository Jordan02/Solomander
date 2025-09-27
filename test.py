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


pd.set_option("display.max_columns", None)


# ==== DATA AND INDICATORS =====
df = s.load_yfinance("MNQ=F", start="2025-08-16", end="2025-09-16", interval="5m")
df['NY'] = s.sessions(df)['NY']
df['vwap'] = s.vwap(df, mode="daily")
df['atr'] = ta.ATR(df, timeperiod=14)


# ==== STRATEGY EXECUTION =====

class strat1(Strategy):

    # ===== SET STRATEGY PARAMETERS =====
    def __init__(self, df : pd.DataFrame, **kwargs):
        super().__init__(df, **kwargs)

        # add variables here, so they can be intellisensed
        self.FEE              = kwargs.get('FEE', 1.74)
        self.LEVERAGE         = kwargs.get('LEVERAGE', 2)
        self.ATR_MULTIPLIER   = kwargs.get('ATR_MULTIPLIER', 1.2)
        self.RR               = kwargs.get('RR', 1.5)
        self.SMA_FAST         = kwargs.get('SMA_FAST', 16)
        self.SMA_SLOW         = kwargs.get('SMA_SLOW', 46)
        self.SIZE             = kwargs.get('SIZE', 1)

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

pnl_curve = []
sr_curve = []

def objective(trial):

    # Define the hyperparameters to optimize
    atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 3.0)
    #rr = trial.suggest_float("rr", 1.0, 3.0)
    sma_fast = trial.suggest_int("sma_fast", 6, 30)
    sma_slow = trial.suggest_int("sma_slow", 40, 60)

    # Create and run the strategy
    st = strat1(  df, 
                  FEE = 1.74, 
                  LEVERAGE = 2,
                  ATR_MULTIPLIER=atr_multiplier, 
                  RR=1.5, 
                  SMA_FAST=sma_fast, 
                  SMA_SLOW=sma_slow)
    
    st.execute()

    pnl_curve.append(st.tf['pnl'].cumsum().to_numpy())
    sr_curve.append(st.SHARPE_RATIO_ANNUAL)

    return st.PNL


#study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
#study.optimize(objective, n_trials=100, n_jobs=1)

#print("Best Sharpe:", study.best_value)
#print("Best Params:", study.best_params)

#see = vis.plot_optimization_history(study)
#see.write_html("opt_history.html", auto_open=True)

#vis.plot_param_importances(study).show()
#vis.plot_parallel_coordinate(study).show()

st = strat1(df,
            FEE=1.74, 
            LEVERAGE = 2,
            ATR_MULTIPLIER=2.2,
            RR=1.5,
            SMA_FAST=9,
            SMA_SLOW=49,
            SIZE=1)

st.execute()

st.show()
st.print_metrics()

s.monte_carlo(st.tf, runs=500, mode='bootstrap', seed=156)
plt.show()
#st.print_metrics()
#tt = s.monte_carlo_data(st.tf, runs=500, mode='bootstrap', seed=42)



