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
TRIAL_TOGGLE = True
NOISE_TEST_TOGGLE = False
MONTE_CARLO_TOGGLE = False
SYMBOL = "US100.cash"


s.mt5_login()
ticker = s.mt5_symbol_info(SYMBOL)
df = s.mt5_hdata(SYMBOL, mt5.TIMEFRAME_M15, candle_lookback=8000)
#df = s.load_yfinance(SYMBOL, start="2025-08-16", end="2025-09-16", interval="5m")
#ticker = s.load_symbol(SYMBOL)


def pid(target, pull, Kp=0.2, Ki=0.05, Kd=0.1,Kv=0.00):

    pull_series = pull.copy().fillna(method='bfill').fillna(method='ffill')
    target_series = target.copy().fillna(method='bfill').fillna(method='ffill')

    n = len(target_series)
    y = np.zeros(n)
    e_prev = 0.0
    integral = 0.0
    pull_error = 0.0

    # initialize PID_signal as first target
    y[0] = target_series.iloc[0]

    for t in range(1, n):
        pull_error = pull_series.iloc[t] - target_series.iloc[t-1]

        error = target_series.iloc[t] - y[t-1]
        integral += error
        derivative = error - e_prev
        e_prev = error

        u = (Kp * error + Ki * integral + Kd * derivative)+ (Kv * pull_error)
        y[t] = y[t-1] + u

    return pd.Series(y, index=target_series.index, name="PID_signal")



class strat2(Backtester):

    # add Input and settings here, so they can be intellisensed
    ATR_MULTIPLIER: float = 1.1
    RR: float = 1.5
    SMA_FAST: int = 6
    SIZE: float = 1
    ALLOWED_OPEN_TRADES: int = 1
    ATR_RANGE: int = 7
    ERROR_UPPER: float = 60
    ERROR_LOWER: float = -60
    ERROR_STATE: int = 0  # 0 = neutral, 1 = above upper, -1 = below lower
    Kp: float = 0.1
    Ki: float = 0.04
    Kd: float = 0.02


    # ===== SET STRATEGY PARAMETERS =====
    def update_data(self):

        # update only those that are dynamic during optimsation, fix others outside
        self.df['SMA_fast'] = ta.SMA(self.df['close'], timeperiod=self.SMA_FAST)
        self.df['vwap'] = s.vwap(self.df, mode='daily')
        #self.df['crossover'] = pta.cross(self.df['SMA_fast'], self.df['SMA_slow'])
        self.df['NY'] = s.sessions(self.df)['NY']
        self.df['atr'] = ta.ATR(self.df, timeperiod=self.ATR_RANGE)
        self.df['pid'] = pid(self.df['SMA_fast'],self.df['vwap'], Kp=self.Kp, Ki=self.Ki, Kd=self.Kd, Kv=0.0)
        self.df['error'] = self.df['SMA_fast'] - self.df['pid'] 
        
        self.df['buy_crossover'] = pta.cross(self.df['error'], pd.Series(self.ERROR_UPPER, index=self.df.index))
        self.df['sell_crossover'] = pta.cross(pd.Series(self.ERROR_LOWER, index=self.df.index),self.df['error'])
        
        return
    
     # ===== Visualization =====

    def plots(self):
        super().plots(rows=4)

        fplt.plot(self.df['SMA_fast'] , ax=self.axs[0], color="#ff6a00", legend=f"SMA {self.SMA_FAST}")
        fplt.plot(self.df['vwap'] , ax=self.axs[0], color="#00ff6a", legend=f"daily VWAP")
        fplt.plot(self.df['atr'], ax=self.axs[1], color="#00ff6a", legend=f"ATR {self.ATR_RANGE}")
        fplt.plot(self.df['error'], ax=self.axs[2], color="#ff00a6", legend="Error")
        fplt.plot(self.df['pid'], ax=self.axs[0], color="#030099da", legend="PID")

        fplt.add_line((self.df.index[0], self.ERROR_UPPER), (self.df.index[-1], self.ERROR_UPPER), ax=self.axs[2], color="#00FF4C", style="--")
        fplt.add_line((self.df.index[0], self.ERROR_LOWER), (self.df.index[-1], self.ERROR_LOWER), ax=self.axs[2], color="#FF4C00", style="--")
        fplt.plot(self.df['buy_crossover']*self.ERROR_UPPER, ax=self.axs[2], color="#007a29da", legend="Signals")
        fplt.plot(self.df['sell_crossover']*self.ERROR_LOWER, ax=self.axs[2], color="#7a0e00da", legend="Signals")

        s.plot_timeband(self.df, 'NY', ax=self.axs[0], color="#a8a8a83d", title="NY")

    # ===== BUY LOGIC =====
    def buy_condition(self, i):
        
        #cond = 1 if self.data['error'][i-1] >

        return 1 if self.data['buy_crossover'][i-1] > 0 else 0

    def buy_action(self, i):
        
        sl = self.data['atr'][i]*self.ATR_MULTIPLIER
        tp = sl * self.RR
        self.buy_bracket(i, self.SIZE, sl_pips=sl, tp_pips=tp, comments='BUY')
        return 
    
    # ===== SELL LOGIC =====
    def sell_condition(self, i):

        return 1 if self.data['sell_crossover'][i-1] > 0 else 0
        
    
    def sell_action(self, i):
    
        sl = self.data['atr'][i]*self.ATR_MULTIPLIER
        tp = sl * self.RR
        self.sell_bracket(i, self.SIZE, sl_pips=sl, tp_pips=tp, comments='SELL')
        return 

   

# ==== OPTIMIZATION =====

pnl_curve = []
display_values = []

def objective(trial):

    # Define the hyperparameters to optimize
    Kp = trial.suggest_float("Kp", 0.01, 1.0)
    Ki = trial.suggest_float("Ki", 0.01, 1.0)
    Kd = trial.suggest_float("Kd", 0.01, 1.0)
    

    

    # Create and run the strategy
    st = strat2(df,
                  ticker,
                  Kp=Kp, 
                  Ki=Ki,
                  Kd=Kd)
    
    st.execute()

    
    pnl_curve.append([] if st.tf.get('pnl', pd.Series()).sum() == 0 else st.tf['pnl'].cumsum().to_numpy())

    display_values.append({  "Kp": Kp, 
                             "Ki": Ki,
                             "Kd": Kd,
                             "mdd": st.MAX_DRAWDOWN, 
                             "sr": st.SHARPE_RATIO_ANNUAL,
                             "pnl": st.PNL,
                        })

    return st.SHARPE_RATIO_ANNUAL

#test_params={"atr_multiplier": 1.83, "sma_fast": 6, "sma_slow": 46, "open_trades": 1, "rr":2.0}


if TRIAL_TOGGLE:
    study = s.Optimise( objective,
                        n_trials=400,
                        n_jobs=1,
                        guess=[],
                        direction = 'maximize',
                        target="SR")
    best_params = study.execute()
    study.show(pnl_curves=pnl_curve, info=display_values)
    test_params = best_params


# ==== EXECUTION =====


st = strat2(df,
            ticker)


st.execute()
print(st.tf)
st.print_metrics()
st.show()
print(st.df['pid'])

if MONTE_CARLO_TOGGLE:
    s.monte_carlo(st.tf, runs=300, mode='bootstrap', seed=156, params=st.INPUT_PARAMS)

if NOISE_TEST_TOGGLE:
    #test_params={'SMA_FAST':1,'SMA_SLOW':1}
    #test_params={'ATR_MULTIPLIER':0.01, 'RR': 0.02}
    s.noise_test(st, test_params=test_params, nudges=3)


plt.show()




