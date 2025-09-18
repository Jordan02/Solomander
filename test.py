import solomander as s
import pandas as pd
from solomander.logger import log, stamp, pront
import finplot as fplt
import talib
import talib.abstract as ta
import pandas_ta as pta
import numpy as np

pd.set_option("display.max_columns", None)

# == constants

SMA_SLOW = 50
SMA_FAST = 20

# == load data
df = s.load_yfinance("MNQ=F", start="2025-08-16", end="2025-09-16", interval="5m")

ax, ax2 = fplt.create_plot('MNQ Chart', rows=2)
fplt.volume_ocv(df[['open', 'close', 'volume']], ax=ax2)
fplt.candlestick_ochl(df, ax = ax) 

# == indicators

#help(talib.SMA)
df['SMA_slow'] = ta.SMA(df, timeperiod=SMA_SLOW)
df['SMA_fast'] = ta.SMA(df, timeperiod=SMA_FAST)
df['crossover'] = pta.cross(df['SMA_fast'], df['SMA_slow'])


fplt.plot(df['SMA_slow'] , ax=ax, color="#ff6a00", legend=f"SMA {SMA_SLOW}")
fplt.plot(df['SMA_fast'] , ax=ax, color="#00ff6a", legend=f"SMA {SMA_FAST}")


s.sessions(df, ax=ax)
s.vwap(df, ax=ax, mode='daily', color="#0c29cf")
s.timeband(df, ax=ax, title="rth", color="#a8a8a830")

# == strategy logic

df['buy_signals'] = pd.Series(index=df.index, dtype='int') # df buy signals
of = pd.DataFrame(columns=['trade_id','time', 'side', 'price','type','qty','filled','activity','comments']) # order df
orders = []


ORDERS = 0
SL = 10 # ticks
TP = 20 # ticks

for i in range(len(df)):
    
    if df.loc[df.index[i],'crossover'] > 0 and ORDERS <= 400: #buy condition

        # raise buy order
        new_order_buy = [{'trade_id': ORDERS,'time': df.index[i], 'side': 'buy', 'price': df.loc[df.index[i],'open'], 'qty': 1, 'filled': 0,'type': 'market', 'comments': '', 'activity': 'open'}]
        
        # set SL
        new_order_sl = [{'trade_id': ORDERS,'time': df.index[i], 'side': 'sell', 'price': df.loc[df.index[i],'open'] - 10, 'qty': 1,'filled': 0, 'type': 'limit', 'comments': 'sl', 'activity': 'open'}]
        

        # set TP
        new_order_tp = [{'trade_id': ORDERS,'time': df.index[i], 'side': 'sell', 'price': df.loc[df.index[i],'open'] + 20, 'qty': 1, 'filled': 0, 'type': 'limit', 'comments': 'tp', 'activity': 'open'}]
        
        orders.append(new_order_buy)
        orders.append(new_order_sl)
        orders.append(new_order_tp)
        # fplt df buy signal
        df.loc[df.index[i],'buy_signals'] = 1 # show buy signal
        ORDERS += 1

    if 1: #sell condition 

        # checking SL and TP conditions
        if not of[of['activity'] == 'open'].empty:

            for _, order in of[of['activity'] == 'open'].iterrows():

                # buy SL
                # add qty and fill to order list i.e. 3/3 filled, when filled order is closed.
                # if a sell order is placed, we check open orders, and see if we can fill any, to allow dynamic closing too
                # if is the case, that order takes trade id of the open order.
                #df.loc[df.index[i],'low'] 

                continue



of = pd.concat([of, pd.DataFrame([item for sublist in orders for item in sublist])], ignore_index=True)


fplt.plot(df.loc[df['buy_signals']>0, 'open'] , ax=ax, color="#000c4d", legend="buys", style="^")
    
stamp.info(f"Total Orders: {df['buy_signals'].sum()}")
#pront.info(df.head())
pront.info(of)

fplt.show()


