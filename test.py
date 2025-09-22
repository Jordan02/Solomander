import solomander as s
import pandas as pd
from solomander.logger import log, stamp, pront
import finplot as fplt
import talib
import talib.abstract as ta
import pandas_ta as pta
import numpy as np

#pd.set_option("display.max_columns", None)

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


#s.sessions(df, ax=ax)
s.vwap(df, ax=ax, mode='daily', color="#0c29cf")
s.timeband(df, ax=ax, title="rth", color="#a8a8a830")

# == strategy logic

orders_open =[]
orders_closed =[]
trades =[]

ORDERS = 0
OPEN_LIMIT_ORDERS = 0
OPEN_TRADES = 0
ORDER_ID = 0
SL = 10 # ticks
TP = 20 # ticks

## BEFORE LOOP CONVERT PANDAS DF TO NUMPY ARRAYS FOR FASTER AND EASIER PROCESSING ## also for order lists
## copy allows orriding of np values, no copying gives readonly arrays
data = {col: df[col].to_numpy().copy() for col in df.columns}
data['datetime'] = df.index.to_numpy().copy()

pront.info(data['datetime'][5])

for i in range(len(df)):
    
    if data['crossover'][i] > 0 and ORDERS <= 100/3: #buy condition

        # raise buy order, SL, TP
        orders_open.append({'trade_id': ORDER_ID,'entry_time': data['datetime'][i], 'side': 'buy', 'price': data['open'][i], 'qty': 1, 'filled': 0,'type': 'market', 'comments': ''})
        orders_open.append({'trade_id': ORDER_ID,'entry_time': data['datetime'][i], 'side': 'sell', 'price': data['open'][i] - 10, 'qty': 1,'filled': 0, 'type': 'sl', 'comments': ''})
        orders_open.append({'trade_id': ORDER_ID,'entry_time': data['datetime'][i], 'side': 'sell', 'price': data['open'][i] + 60, 'qty': 1, 'filled': 0, 'type': 'tp', 'comments': ''})
        
        # fplt df buy signal

        ORDERS += 3
        ORDER_ID += 1
        OPEN_LIMIT_ORDERS +=2
        OPEN_TRADES +=1

    if 1: #sell condition 

        if OPEN_LIMIT_ORDERS > 0:

            # for each open order
            order_market = [o for o in orders_open if o['type'] == 'market']

            for order in order_market:

                trade_id = order['trade_id'] #retreive trade id VALUE
                order_sl = next(o for o in orders_open if o['trade_id']==trade_id and o['type'] =='sl') # return List of sl orders for order id
                order_tp = next(o for o in orders_open if o['trade_id']==trade_id and o['type'] =='tp') # return List of tp orders for order id
                
                ## ==== SL HIT ====
                if data['high'][i] >= order_tp['price']: 

                    # add trade order record
                    pnl = order_tp['price'] - order['price']
                    trades.append({ 'trade_id': trade_id,
                                    'entry_time': order['entry_time'],
                                    'exit_time': data['datetime'][i], 
                                    'entry_price': order['price'], 
                                    'exit_price': order_tp['price'], 
                                    'side': 'sell', 
                                    'qty': 1, 
                                    'filled': 1, 
                                    'sl': order_sl['price'], 
                                    'tp': order_tp['price'], 
                                    'pnl':pnl, 
                                    'comments': ''})
                    
                    #move remove from open_orders
                    orders_open.remove(order)
                    orders_open.remove(order_sl)
                    orders_open.remove(order_tp)

                    # add to closed_orders
                    order['filled'] = 1
                    order_sl['comments'] = 'cancelled'
                    order_tp['filled'] = 1
                    orders_closed.append(order)
                    orders_closed.append(order_sl)
                    orders_closed.append(order_tp)

                    # update signals and counters
                    OPEN_LIMIT_ORDERS -=2
                    OPEN_TRADES -=1
                    continue
                
                ## ==== SL HIT ====
                if data['low'][i] <= order_sl['price']:

                    # add trade order record
                    pnl = order['price'] - order_sl['price']
                    trades.append({ 'trade_id': trade_id,
                                    'entry_time': order['entry_time'],
                                    'exit_time': data['datetime'][i], 
                                    'entry_price': order['price'], 
                                    'exit_price': order_sl['price'], 
                                    'side': 'sell', 
                                    'qty': 1, 
                                    'filled': 1, 
                                    'sl': order_sl['price'], 
                                    'tp': order_tp['price'], 
                                    'pnl':pnl, 
                                    'comments': ''})
                    
                    #move remove from open_orders
                    orders_open.remove(order)
                    orders_open.remove(order_sl)
                    orders_open.remove(order_tp)

                    # add to closed_orders
                    order['filled'] = 1
                    order_tp['comments'] = 'cancelled'
                    order_sl['filled'] = 1
                    orders_closed.append(order)
                    orders_closed.append(order_sl)
                    orders_closed.append(order_tp)

                    # update signals and counters
                    OPEN_LIMIT_ORDERS -=2
                    OPEN_TRADES -=1
                    continue

            
                


## recreate pandas from vectors
df = pd.DataFrame(data, index=data['datetime'])
pd_orders_open = pd.DataFrame(orders_open)
pd_orders_closed = pd.DataFrame(orders_closed)  
tf = pd.DataFrame(trades)

pront.critical(tf)

pront.info(pd_orders_open)
pront.debug(pd_orders_closed)
pront.warning(tf)

s.plot_trades(tf=tf, df=df, ax=ax)

fplt.show()


