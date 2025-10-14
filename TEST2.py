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



DISCORD_REPORT = False

OPTIMISE = False
OPTIMISE_TRIALS = 500
COLOR_OPTIMISE = "#b3ff00"

MONTE_CARLO = True
MONTE_CARLO_RUNS = 300
MONTE_CARLO_MODE = "bootstrap"  # permutation / bootstrap
MONTE_CARLO_COLOR = "#04C8EB"

NOISE_TEST = False
NOISE_TEST_NUDGES = 5
NOISE_TEST_PARAMS={'SMA_FAST':1, 'SMA_SLOW':1}
NOISE_TEST_COLOR = "#fffb00"

SHOW_CANDLE_CHART = True
SYMBOL = "US100.cash"
TIME_FRAME = mt5.TIMEFRAME_M5
CANDLE_LOOKBACK = 4000




if DISCORD_REPORT:
    matplotlib.use("Agg")  
    bot = s.DiscordBot()
    bot.am_ready.wait()  #wait till bot is ready
else:
    matplotlib.use("TkAgg")

pd.set_option("display.max_columns", None)

s.mt5_login()
ticker = s.mt5_symbol_info(SYMBOL)
df = s.mt5_hdata(SYMBOL, TIME_FRAME, candle_lookback=CANDLE_LOOKBACK)


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
    st = strat1( 
                  ATR_MULTIPLIER=atr_multiplier, 
                  RR=rr, 
                  SMA_FAST=sma_fast, 
                  SMA_SLOW=sma_slow,
                  ALLOWED_OPEN_TRADES=open_trades,
                  SIZE=1 )
    
    
    study = Backtester(st,df,ticker)
    result = study.execute()
    
    pnl_curve.append(result.l_CUMSUM_PNL)

    display_values.append({  "atr_multiplier": atr_multiplier, 
                             "sma_fast": sma_fast,
                             "sma_slow": sma_slow, 
                             "mdd": result.TOTAL_MAX_DRAWDOWN, 
                             "sr": result.TOTAL_SHARPE_RATIO_ANNUAL,
                             "pnl": result.TOTAL_PNL,
                             "open_trades": open_trades,
                             "rr": rr})

    return result.TOTAL_SHARPE_RATIO_ANNUAL

#guess_1={"atr_multiplier": 1.83, "sma_fast": 6, "sma_slow": 46}
guess_2={"atr_multiplier": 2.67, "sma_fast": 23, "sma_slow": 127, "rr": 2.76, "open_trades": 1}
guess_3={"atr_multiplier": 1.62, "sma_fast": 17, "sma_slow": 55, "rr": 1.5, "open_trades": 3}
guess_4={"atr_multiplier": 2.35, "sma_fast": 25, "sma_slow": 58, "rr": 1.78, "open_trades": 3}
guess_5={"atr_multiplier": 2.07, "sma_fast": 17, "sma_slow": 56, "rr": 2.17, "open_trades": 1}
guess_11={"atr_multiplier": 1.44, "sma_fast": 18, "sma_slow": 108, "rr": 1.68, "open_trades": 8}


if OPTIMISE:
    study = s.Optimise( objective,
                        n_trials=OPTIMISE_TRIALS,
                        n_jobs=1,
                        guess=[],
                        direction = 'maximize',
                        target="SR")
    best_params = study.execute()

    buf = study.show(pnl_curves=pnl_curve, info=display_values, discord=DISCORD_REPORT, color=COLOR_OPTIMISE)

    if DISCORD_REPORT:
        bot.post_fig(buf, "📈 Optimization Results", color=COLOR_OPTIMISE)

else:

    best_params={"atr_multiplier": 2, "sma_fast": 17, "sma_slow": 54, "rr": 1.3, "open_trades": 4}

# ==== EXECUTION =====


st = strat1(
            ATR_MULTIPLIER= best_params['atr_multiplier'],
            RR=best_params['rr'],
            SMA_FAST= best_params['sma_fast'],
            SMA_SLOW=best_params['sma_slow'],
            ALLOWED_OPEN_TRADES= best_params['open_trades'],
            SIZE=1)

study = Backtester(st,df,ticker)
result = study.execute()

if SHOW_CANDLE_CHART:
    study.show()


study.print_metrics()
print(result.tf.head(20))

if MONTE_CARLO:
    buf = s.monte_carlo(result, runs=MONTE_CARLO_RUNS, mode=MONTE_CARLO_MODE, discord=DISCORD_REPORT, color=MONTE_CARLO_COLOR, seed=156, params=st.INPUT_PARAMS)

    if DISCORD_REPORT:
        bot.post_fig(buf, "📊 Monte Carlo Results", color=MONTE_CARLO_COLOR)


if NOISE_TEST:
    buf = s.noise_test(result, test_params=NOISE_TEST_PARAMS, nudges=NOISE_TEST_NUDGES, color=NOISE_TEST_COLOR, discord=DISCORD_REPORT)

    if DISCORD_REPORT:
        
        bot.post_fig(buf, "📢 Noise Test Results", color=NOISE_TEST_COLOR)
    


plt.show()




