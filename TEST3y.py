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


DISCORD_REPORT = True
DISCORD_CHANNEL_ID = 1427792207930458273  

OPTIMISE = False
OPTIMISE_TRIALS = 200
COLOR_OPTIMISE = "#ff7b00"

MONTE_CARLO = False
MONTE_CARLO_RUNS = 300
MONTE_CARLO_MODE = "bootstrap"  # permutation / bootstrap
MONTE_CARLO_COLOR = "#04C8EB"

NOISE_TEST = False
NOISE_TEST_NUDGES = 10
#NOISE_TEST_PARAMS={'SMA_FAST':1, 'SMA_SLOW':1}
NOISE_TEST_PARAMS={'SMA_SLOW':1}
NOISE_TEST_COLOR = "#fffb00"

SHOW_CANDLE_CHART = True
SYMBOL = "US100.cash"
TIME_FRAME = mt5.TIMEFRAME_M15
CANDLE_LOOKBACK = 8000
CANDLE_OFFSET = 0  # to shift candles back in time for testing
#DEFAULT_PARAMS = {"atr_multiplier": 2.97, "sma_fast": 26, "sma_slow": 135, "rr": 2.77, "open_trades": 1}
DEFAULT_PARAMS = {"atr_multiplier": 2.72, "sma_fast": 55, "sma_slow": 233, "rr": 1.2, "open_trades": 4}
ORDER_SIZE = 1.0  # lots

TEST_MESSAGE = "Test script starting..."


if DISCORD_REPORT:
    matplotlib.use("Agg")  
    bot = s.DiscordBot()
    bot.am_ready.wait()  #wait till bot is ready
    bot.post_message(TEST_MESSAGE, channel_id=DISCORD_CHANNEL_ID)
else:
    matplotlib.use("TkAgg")

pd.set_option("display.max_columns", None)

s.mt5_login()
ticker = s.mt5_load_symbol(SYMBOL)
df = s.mt5_hdata(SYMBOL, TIME_FRAME, candle_lookback=CANDLE_LOOKBACK, candle_offset=CANDLE_OFFSET)


class strat1(Strategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ATR_PERIOD = kwargs.get("ATR_PERIOD", 14)
        self.LOOKBACK = kwargs.get("LOOKBACK", 20)
        self.ATR_LOOKBACK = kwargs.get("ATR_LOOKBACK", 100)
        self.ATR_PERCENTILE = kwargs.get("ATR_PERCENTILE", 0.2)  # 20%
        self.ATR_MULTIPLIER = kwargs.get("ATR_MULTIPLIER", 1.5)
        self.SIZE = kwargs.get("SIZE", 1.0)
        self.ALLOWED_OPEN_TRADES = kwargs.get("ALLOWED_OPEN_TRADES", 1)
        self.RR = kwargs.get("RR", 2.0)

    # ===== UPDATE INDICATORS =====
    def update_data(self):
        df = self.df

        df["atr"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=self.ATR_PERIOD)
        df["atr_rank"] = df["atr"].rolling(self.ATR_LOOKBACK).rank(pct=True)
        df["low_vol"] = df["atr_rank"] < self.ATR_PERCENTILE

        df["range_high"] = df["high"].rolling(self.LOOKBACK).max().shift(1)
        df["range_low"] = df["low"].rolling(self.LOOKBACK).min().shift(1)

        df["breakout_up"] = (df["close"] > df["range_high"]) & df["low_vol"]
        df["breakout_dn"] = (df["close"] < df["range_low"]) & df["low_vol"]

        df["NY"] = s.sessions(df)["NY"]
        return

    # ===== BUY LOGIC =====
    def buy_condition(self, i):
        return (
            self.data["breakout_up"][i-1]
            and self.OPEN_TRADES < self.ALLOWED_OPEN_TRADES
            and self.data["NY"][i-1] > 0
        )

    def buy_action(self, i):
        sl = self.data["atr"][i] * self.ATR_MULTIPLIER
        tp = sl * self.RR
        return self.buy_bracket(i, self.SIZE, sl_pips=sl, tp_pips=tp, comments="VolBreak")

    # ===== SELL LOGIC =====
    def sell_condition(self, i):
        return (
            self.data["breakout_dn"][i-1]
            and self.OPEN_TRADES < self.ALLOWED_OPEN_TRADES
            and self.data["NY"][i-1] > 0
        )

    def sell_action(self, i):
        sl = self.data["atr"][i] * self.ATR_MULTIPLIER
        tp = sl * self.RR
        return self.sell_bracket(i, self.SIZE, sl_pips=sl, tp_pips=tp, comments="VolBreak")

    # ===== VISUALIZATION =====
    def _plots(self):
        super()._plots(rows=3)
        fplt.plot(self.df["atr"], ax=self.axs[1], color="#ffaa00", legend="ATR")
        fplt.plot(self.df["range_high"], ax=self.axs[0], color="#ff6a00", style="--", legend="Range High")
        fplt.plot(self.df["range_low"], ax=self.axs[0], color="#00ff6a", style="--", legend="Range Low")
        fplt.plot(self.df["close"], ax=self.axs[0], color="#ffffff", legend="Close")

        s.plot_timeband(self.df, "NY", ax=self.axs[0], color="#a8a8a83d", title="NY")


# ==== OPTIMIZATION =====

pnl_curve = []
display_values = []

def objective(trial):

    # Define the hyperparameters to optimize
    atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 3.0)
    rr = trial.suggest_float("rr", 1.0, 2.0)
    sma_fast = trial.suggest_int("sma_fast", 6, 60)
    sma_slow = trial.suggest_int("sma_slow", 40, 250)
    open_trades = trial.suggest_int("open_trades", 1, 5)

    # Create and run the strategy
    st = strat1( 
                  ATR_MULTIPLIER=atr_multiplier, 
                  RR=rr, 
                  SMA_FAST=sma_fast, 
                  SMA_SLOW=sma_slow,
                  ALLOWED_OPEN_TRADES=open_trades,
                  SIZE=ORDER_SIZE )
    
    
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

guess_1={"atr_multiplier": 2.97, "sma_fast": 26, "sma_slow": 135, "rr": 2.77, "open_trades": 1}


if OPTIMISE:
    study = s.Optimise( objective,
                        n_trials=OPTIMISE_TRIALS,
                        n_jobs=1,
                        guess=[DEFAULT_PARAMS],
                        direction = 'maximize',
                        target="SR")
    best_params = study.execute()
else:

    best_params=DEFAULT_PARAMS

# ==== EXECUTION =====


st = strat1(
            ATR_MULTIPLIER= best_params['atr_multiplier'],
            RR=best_params['rr'],
            SMA_FAST= best_params['sma_fast'],
            SMA_SLOW=best_params['sma_slow'],
            ALLOWED_OPEN_TRADES= best_params['open_trades'],
            SIZE=ORDER_SIZE)

study = Backtester(st,df,ticker)
result = study.execute()


if DISCORD_REPORT:
    bot.post_embed(result.discord_summary(), channel_id=DISCORD_CHANNEL_ID)

if SHOW_CANDLE_CHART:
    study.show()
    
if OPTIMISE:
        opt_buf = study.show(pnl_curves=pnl_curve, info=display_values, discord=DISCORD_REPORT, color=COLOR_OPTIMISE)
        
        if DISCORD_REPORT:
            bot.post_fig(opt_buf, "📈 Optimization Results", color=COLOR_OPTIMISE, channel_id = DISCORD_CHANNEL_ID)

if MONTE_CARLO:
    buf = s.monte_carlo(result, runs=MONTE_CARLO_RUNS, mode=MONTE_CARLO_MODE, discord=DISCORD_REPORT, color=MONTE_CARLO_COLOR, seed=156, params=st.INPUT_PARAMS)

    if DISCORD_REPORT:
        bot.post_fig(buf, "📊 Monte Carlo Results", color=MONTE_CARLO_COLOR, channel_id = DISCORD_CHANNEL_ID)


if NOISE_TEST:
    buf = s.noise_test(result, test_params=NOISE_TEST_PARAMS, nudges=NOISE_TEST_NUDGES, color=NOISE_TEST_COLOR, discord=DISCORD_REPORT)

    if DISCORD_REPORT:
        
        bot.post_fig(buf, "📢 Noise Test Results", color=NOISE_TEST_COLOR, channel_id = DISCORD_CHANNEL_ID)


plt.show()




