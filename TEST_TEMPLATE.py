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


# ========================================
# CONFIGURATION
# ========================================

# Discord Settings
DISCORD_REPORT = True
DISCORD_CHANNEL_ID = 1427792207930458273

# Pipeline Controls
OPTIMISE = False
OPTIMISE_TRIALS = 200

MONTE_CARLO = False
MONTE_CARLO_RUNS = 300
MONTE_CARLO_MODE = "bootstrap"  # "permutation" or "bootstrap"

NOISE_TEST = False
NOISE_TEST_NUDGES = 10
NOISE_TEST_PARAMS = {'MIN_RANGE_ATR': 0.05, 'ATR_MULTIPLIER': 0.1}

SHOW_CANDLE_CHART = True

# Market Settings
SYMBOL = "US100.cash"
#SYMBOL = "MNQ=F"
TIME_FRAME = mt5.TIMEFRAME_M15  # M5, M15, H1
CANDLE_LOOKBACK = 4000
CANDLE_OFFSET = 0

# Strategy Parameters
DEFAULT_PARAMS = {
    "OR_PERIOD": 1,           # 1 hour opening range on H1 bars
    "MIN_RANGE_ATR": 0.15,    
    "ATR_MULTIPLIER": 2.0,    
    "RR": 2.0,                
}
ORDER_SIZE = 1.0  # Lot size

# Visual Colors
COLOR_OPTIMISE = "#3498db"
MONTE_CARLO_COLOR = "#04C8EB"
NOISE_TEST_COLOR = "#fffb00"


# ========================================
# SETUP
# ========================================

if DISCORD_REPORT:
    matplotlib.use("Agg")  
    bot = s.DiscordBot()
    bot.am_ready.wait()
else:
    matplotlib.use("TkAgg")

pd.set_option("display.max_columns", None)

s.mt5_login()
df = s.mt5_hdata(SYMBOL, TIME_FRAME, candle_lookback=CANDLE_LOOKBACK, candle_offset=CANDLE_OFFSET, download_data =False)
ticker = s.mt5_load_symbol(SYMBOL, read = False, write = True)


#df = s.yfin_load_data(SYMBOL,start = "2025-10-02", end = "2025-10-16", interval = "30m",read=True, write=True)
#ticker = s.read_symbol(SYMBOL)


# ========================================
# STRATEGY CLASS
# ========================================

class OpeningRangeBreakout(Strategy):
    """
    Opening Range Breakout (ORB)
    - Define opening range as the first `OR_PERIOD` NY-session bars each day.
    - Go long when price **crosses above** OR high after the OR is complete.
    - Go short when price **crosses below** OR low after the OR is complete.
    - One trade per day; ATR-based SL/TP.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.OR_PERIOD       = int(kwargs.get("OR_PERIOD", 4))      # first 4 NY bars (4h on 1H chart)
        self.MIN_RANGE_ATR   = kwargs.get("MIN_RANGE_ATR", 0.15)    # OR must be >= 0.15 * ATR
        self.ATR_PERIOD      = int(kwargs.get("ATR_PERIOD", 14))
        self.ATR_MULTIPLIER  = kwargs.get("ATR_MULTIPLIER", 4)
        self.RR              = kwargs.get("RR", 2.0)
        self.SIZE            = kwargs.get("SIZE", 1.0)
        self.ALLOWED_OPEN_TRADES = kwargs.get("ALLOWED_OPEN_TRADES", 1)
        self.EMA_fast = kwargs.get("EMA_fast",3)

        self._last_trade_day = None  # stores date() of last trade
        self.setting_plot_rows = 3

        
    # ===== indicators / state =====
    def update_data(self):

        self.atr = f"atr_{self.ATR_PERIOD}"
        self.ema = f"ema_{self.EMA_fast}"
        self.vwap = "vwap"
        self.in_ny = "NY"
        self.in_asia = "TKY"
        self.in_london = "LND"
        self.asia_highs = "asia_highs"
        self.asia_lows = "asia_lows"
        self.london_highs = "london_highs"
        self.london_lows = "london_lows"
        self.cross_high = "cross_high"

        self.df[self.atr] = ta.ATR(self.df, timeperiod = self.ATR_PERIOD)
        self.df[self.ema] = ta.SMA(self.df, timeperiod = self.EMA_fast)
        self.df[self.vwap] = s.vwap(self.df, mode = "daily")

        sessions = s.sessions(self.df)
        sessions_value_highs = s.sessions_value(self.df,'high', agg='max', mode="extend")
        sessions_value_lows = s.sessions_value(self.df,'low', agg='min', mode="extend")

        self.df[self.in_ny] = sessions['NY']
        self.df[self.in_asia] = sessions['TKY']
        self.df[self.in_london] = s.timeblock(df, start="7:30", end="13:30", tz="UTC")

        self.df[self.asia_highs] = sessions_value_highs['TKY']
        self.df[self.asia_lows] = sessions_value_lows['TKY']
        #self.df["LDN_highs"] = sessions_value_highs['LDN']
        #self.df["LDN_lows"] = sessions_value_lows['LDN']
        self.df[self.london_highs] = s.timeblock_value(df, value="high", agg="max", start="7:30",  end="13:30", tz="UTC", mode="extend")
        self.df[self.london_lows] = s.timeblock_value(df, value="high", agg="max", start="7:30",  end="13:30", tz="UTC", mode="extend")
        
        self.df["day"] = self.df.index.date

        self.df[self.cross_high] = pta.cross(self.df[self.ema], self.df[self.asia_highs])

        


     # ===== plots =====
    def plots(self):
        
        # ==== graph 1 ===
        s.plot_timeblock(self.df, self.in_ny, ax = self.axs[0], color="#e9905462", title=self.in_ny )
        #s.plot_timeblock(self.df, "in_LDN",ax = self.axs[0], color="#54e96d62", title="LDN" )
        s.plot_timeblock(self.df, self.in_london,ax = self.axs[0], color="#54e96d62", title=self.in_london )
        s.plot_timeblock(self.df, self.in_asia, ax = self.axs[0], color="#d854e962", title=self.in_asia)
        
        fplt.plot(self.df[self.asia_highs], ax=self.axs[0], color="#b301fa", legend=self.asia_highs)
        fplt.plot(self.df[self.asia_lows], ax=self.axs[0], color="#6d005e", legend=self.asia_lows)
        #fplt.plot(self.df['LDN_highs'], ax=self.axs[0], color="#01d641", legend="LDN_highs")
        #fplt.plot(self.df['LDN_lows'], ax=self.axs[0], color="#036d08", legend="LDN_lows")
        fplt.plot(self.df[self.london_highs], ax=self.axs[0], color="#5bb607", legend=self.london_highs)
        fplt.plot(self.df[self.london_lows], ax=self.axs[0], color="#036d08", legend=self.london_lows)
        
        fplt.plot(self.df[self.vwap], ax=self.axs[0], color="#0004ff", legend=self.vwap, style = '--', width=2)
        fplt.plot(self.df[self.ema], ax=self.axs[0], color="#ff0800", legend=self.ema, width=1)

        # === graph 2 ====
        fplt.plot(self.df[self.atr], ax=self.axs[1], color="#e74c3c", legend=self.atr)

    # ===== BUY =====

    def buy_condition(self, i):
       
        #return self.data[self.cross_high][i-1] and self.data[self.in_london][i-1] > 0 and self.OPEN_TRADES < self.ALLOWED_OPEN_TRADES
        return self.data[self.in_ny][i-2] > 0 and self.data[self.in_ny][i-3] < 0
        

    def buy_action(self, i):
        self._last_trade_day = pd.Timestamp(self.data['datetime'][i]).date()
        sl = float(self.data[self.atr][i]) * float(self.ATR_MULTIPLIER)
        tp = sl * float(self.RR)
        return self.buy_bracket(i, self.SIZE, sl_pips=sl, tp_pips=tp, comments="ORB_L")

    # ===== SELL =====
    def sell_condition(self, i):
        
        return False

    def sell_action(self, i):
        self._last_trade_day = pd.Timestamp(self.data['datetime'][i]).date()
        sl = float(self.data["atr"][i]) * float(self.ATR_MULTIPLIER)
        tp = sl * float(self.RR)
        return self.sell_bracket(i, self.SIZE, sl_pips=sl, tp_pips=tp, comments="ORB_S")

   

      
    


# ========================================
# OPTIMIZATION
# ========================================

pnl_curves = []
display_values = []

def objective(trial):
    """Optimization objective function"""
    
    or_period = trial.suggest_int("OR_PERIOD", 2, 8)
    min_range_atr = trial.suggest_float("MIN_RANGE_ATR", 0.05, 0.5)
    atr_multiplier = trial.suggest_float("ATR_MULTIPLIER", 1.0, 3.5)
    rr = trial.suggest_float("RR", 1.0, 3.0)
    
    st = OpeningRangeBreakout(
        OR_PERIOD=or_period,
        MIN_RANGE_ATR=min_range_atr,
        ATR_MULTIPLIER=atr_multiplier,
        RR=rr,
        SIZE=ORDER_SIZE
    )
    
    study = Backtester(st, df, ticker)
    result = study.execute()
    
    pnl_curves.append(result.l_CUMSUM_PNL)
    display_values.append({
        "OR_PERIOD": or_period,
        "MIN_RANGE_ATR": min_range_atr,
        "ATR_MULTIPLIER": atr_multiplier,
        "RR": rr,
        "SR": result.TOTAL_SHARPE_RATIO_ANNUAL,
        "PNL": result.TOTAL_PNL,
        "MDD": result.TOTAL_MAX_DRAWDOWN
    })
    
    return result.TOTAL_SHARPE_RATIO_ANNUAL


if OPTIMISE:
    stamp.info("🔧 Running optimization...")
    optimizer = s.Optimise(
        objective,
        n_trials=OPTIMISE_TRIALS,
        n_jobs=1,
        guess=[DEFAULT_PARAMS],
        direction='maximize',
        target="Sharpe Ratio"
    )
    best_params = optimizer.execute()
    stamp.success(f"✅ Optimization complete. Best params: {best_params}")
else:
    best_params = DEFAULT_PARAMS
    stamp.info(f"📋 Using default params: {best_params}")


# ========================================
# EXECUTION WITH BEST PARAMS
# ========================================

strategy = OpeningRangeBreakout(
    setting_plot_tradeid=True,
    OR_PERIOD=best_params['OR_PERIOD'],
    MIN_RANGE_ATR=best_params['MIN_RANGE_ATR'],
    ATR_MULTIPLIER=best_params['ATR_MULTIPLIER'],
    RR=best_params['RR'],
    SIZE=ORDER_SIZE
)

study = Backtester(strategy, df, ticker)
result = study.execute()
study.print_metrics()
print(result.df.head(50))
print(result.df.index.tz)


# ========================================
# DISCORD REPORTING
# ========================================

if DISCORD_REPORT:
    bot.post_message("TEST RESULTS", channel_id=DISCORD_CHANNEL_ID)
    #bot.post_data(result.df.tail(400), channel_id=DISCORD_CHANNEL_ID)
    bot.post_embed(result.discord_market(), channel_id=DISCORD_CHANNEL_ID)
    bot.post_embed(result.discord_settings(), channel_id=DISCORD_CHANNEL_ID)
    bot.post_embed(result.discord_stats(), channel_id=DISCORD_CHANNEL_ID)


# ========================================
# OPTIMIZATION RESULTS
# ========================================

if OPTIMISE:
    stamp.info("📊 Generating optimization report...")
    opt_buf = optimizer.show(pnl_curves=pnl_curves, info=display_values, 
                            discord=DISCORD_REPORT, color=COLOR_OPTIMISE)
    
    if DISCORD_REPORT:
        bot.post_fig(opt_buf, "📈 Optimization Results", 
                    color=COLOR_OPTIMISE, channel_id=DISCORD_CHANNEL_ID)


# ========================================
# MONTE CARLO VALIDATION
# ========================================

if MONTE_CARLO:
    stamp.info("🎲 Running Monte Carlo simulation...")
    mc_buf = s.monte_carlo(
        result, 
        runs=MONTE_CARLO_RUNS, 
        mode=MONTE_CARLO_MODE, 
        discord=DISCORD_REPORT, 
        color=MONTE_CARLO_COLOR, 
        seed=42,
        params=strategy.INPUT_PARAMS
    )
    
    if DISCORD_REPORT:
        bot.post_fig(mc_buf, "📊 Monte Carlo Results", 
                    color=MONTE_CARLO_COLOR, channel_id=DISCORD_CHANNEL_ID)


# ========================================
# NOISE TEST VALIDATION
# ========================================

if NOISE_TEST:
    stamp.info("🔊 Running noise test...")
    noise_buf = s.noise_test(
        result, 
        test_params=NOISE_TEST_PARAMS, 
        nudges=NOISE_TEST_NUDGES, 
        color=NOISE_TEST_COLOR, 
        discord=DISCORD_REPORT
    )
    
    if DISCORD_REPORT:
        bot.post_fig(noise_buf, "📢 Noise Test Results", 
                    color=NOISE_TEST_COLOR, channel_id=DISCORD_CHANNEL_ID)


# ========================================
# SHOW CHART
# ========================================

if SHOW_CANDLE_CHART:
    stamp.info("📈 Displaying chart...")
    study.show()

plt.show()