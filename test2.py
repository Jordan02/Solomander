import solomander as s
from solomander.logger import log, stamp, pront
from solomander.strategy import Strategy
from matplotlib import pyplot as plt

import pandas as pd
import finplot as fplt
import talib
import talib.abstract as ta
import pandas_ta as pta
import numpy as np

pd.set_option("display.max_columns", None)

# ==== CONSTANTS =====

SMA_SLOW = 50
SMA_FAST = 20
RR = 2
ATR_MULTIPLIER = 2

# ==== DATA AND INDICATORS =====
df = s.load_yfinance("MNQ=F", start="2025-08-16", end="2025-09-16", interval="5m")
#df = s.load_yfinance("MNQ=F", start="2025-02-16", end="2025-09-16", interval="1h")

#help(talib.SMA)
df['SMA_slow'] = ta.SMA(df, timeperiod=SMA_SLOW)
df['SMA_fast'] = ta.SMA(df, timeperiod=SMA_FAST)
df['crossover'] = pta.cross(df['SMA_fast'], df['SMA_slow'])
df['crossunder'] = pta.cross(df['SMA_slow'], df['SMA_fast'])
df['NY'] = s.sessions(df)['NY']
df['vwap'] = s.vwap(df, mode="daily")
df['atr'] = ta.ATR(df, timeperiod=14)

# ==== STRATEGY EXECUTION =====

class strat1(Strategy):

    # ===== BUY LOGIC =====
    def buy_condition(self, i):
        time_cond = self.data['NY'][i] > 0  # Only trade in NY session
        price_above_vwap = self.data['close'][i] > self.data['vwap'][i]
        pullback = self.data['low'][i] <= self.data['vwap'][i]  # touched VWAP
        return time_cond and price_above_vwap and pullback and self.OPEN_TRADES < 3


    def buy_action(self, i):
        pre_i = max(0, i)
        rr = RR
        sl = self.data['atr'][pre_i] * ATR_MULTIPLIER
        tp = sl * rr
        self.bracket_order(i, 'buy', 1, sl_pips=sl, tp_pips=tp, comments='VWAP-BUY')
        return


    def sell_condition(self, i):
        time_cond = self.data['NY'][i] > 0  # Only trade in NY session
        price_below_vwap = self.data['close'][i] < self.data['vwap'][i]
        pullback = self.data['high'][i] >= self.data['vwap'][i]  # touched VWAP
        return time_cond and price_below_vwap and pullback and self.OPEN_TRADES < 3


    def sell_action(self, i):
        pre_i = max(0, i)
        rr = RR
        sl = self.data['atr'][pre_i] * ATR_MULTIPLIER
        tp = sl * rr
        self.bracket_order(i, 'sell', 1, sl_pips=sl, tp_pips=tp, comments='VWAP-SELL')
        return
    
    
st = strat1(df)
st.FEE = 1.74
st.LEVERAGE = 2

st.execute()

# ==== PLOTTING VISUALS =====

ax, ax3, ax2 = fplt.create_plot('MNQ Chart', rows=3)
fplt.volume_ocv(df[['open', 'close', 'volume']], ax=ax.overlay())
fplt.candlestick_ochl(df, ax = ax) 
fplt.plot(df['atr'], ax=ax3, color="#00ff6a", legend="ATR")

fplt.add_line((df.index[0], st.STARTING_MARGIN), (df.index[-1], st.STARTING_MARGIN), ax = ax2, color="#130000", style="--")
fplt.plot(st.CUM_MARGIN, ax=ax2, color="#ff6a00", legend="cumulative Pnl")

fplt.plot(df['SMA_slow'] , ax=ax, color="#ff6a00", legend=f"SMA {SMA_SLOW}")
fplt.plot(df['SMA_fast'] , ax=ax, color="#00ff6a", legend=f"SMA {SMA_FAST}")
fplt.plot(df['vwap'], ax=ax, color="#219bec", legend="VWAP")

s.plot_timeband(df, 'NY', ax=ax, color="#a8a8a83d", title="NY")
s.plot_trades(tf=st.tf, cc=st.cc, df=df, ax=ax, boxes=True)
   
pront.info(st.tf.head(20))
pront.info(st.CUM_PNL.head(20))
    
# ==== OUTPUTS =====

cum_pnl = st.CUM_PNL['cum_pnl']
pnl = st.tf['pnl']

max_dd = s.max_drawdown(pnl)
pront.info(f"Max Drawdown: {max_dd}")

fig, ax = plt.subplots(figsize=(10,5))

# Plot dots
ax.plot(cum_pnl.index, cum_pnl.values, 'o', color="#ff6a00", markersize=4)
ax.plot(cum_pnl.index, cum_pnl.values, '--', color="#ff6a00", label="cumulative PnL")
ax.axhline(0, color="#130000", linestyle="--")

# Labels & legend
ax.set_title("Monte Carlo")
ax.set_xlabel("Trades")
ax.set_ylabel("PnL")
ax.legend()

st.print_metrics()

fplt.show()
plt.show()


