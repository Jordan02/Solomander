import finplot as fplt
import talib
import talib.abstract as ta
import pandas_ta as pta
import pandas as pd

from typing import final

try:
    from .indicators import vwap, timeband
    from .logger import log, stamp, pront
    from .data import load_yfinance
    from .visuals import plot_trades
except ImportError:
    from indicators import vwap, timeband
    from logger import log, stamp, pront
    from data import load_yfinance
    from visuals import plot_trades
    

class Strategy:
    def __init__(self, df : pd.DataFrame):
        
        # panda dataframes
        self.df = df                   # main dataframe (candles/indiciators)
        self.tf = pd.DataFrame()       # trade dataframe
        self.oo = pd.DataFrame()       # open orders dataframe
        self.cc = pd.DataFrame()       # closed orders dataframe

        # Python lists 
        # for dict navigiation, and faster processing (pandas is slow for this)
        self.orders_open =[]
        self.orders_closed =[]
        self.trades =[]

        # convert df to numpy arrays for faster processing. 
        self.data = {col: df[col].to_numpy().copy() for col in df.columns}
        self.data['datetime'] = df.index.to_numpy().copy() # Copy allows overriding of values

        # signals and counters
        self.ORDER_ID = 0

        self.TOTAL_ORDERS = 0
        self.OPEN_ORDERS = 0
        self.OPEN_LIMIT_ORDERS = 0

        self.TOTAL_TRADES = 0
        self.OPEN_TRADES = 0

    def buy_condition(self, i):
        
        
        # default strategy: crossover of 9 and 21 EMA
        # inherit and override this method for custom strategy
        return self.OPEN_TRADES < 1
        
        
    
    def sell_condition(self, i):
        
        # default strategy: crossover of 9 and 21 EMA
        # inherit and override this method for custom strategy
        #return self.data['crossunder'][i] > 0 and self.TOTAL_TRADES < 2
        return 0

    def raise_order(self, i, side, qty, price=None, type='market', comments=''):   
        
        # TODO account for slippage here.

        # price will default market price if None
        if price is None:
            price = self.data['open'][i]

        self.orders_open.append({'trade_id': self.ORDER_ID,
                                 'entry_time': self.data['datetime'][i], 
                                 'side': side, 
                                 'price': price, 
                                 'qty': qty, 
                                 'filled': 0,
                                 'type': type, 
                                 'comments': comments})
        
        self.ORDER_ID += 1
        self.OPEN_ORDERS +=1
        self.TOTAL_TRADES +=1
        self.TOTAL_ORDERS +=1

        return
    
    @final
    def bracket_order(self, i, side, qty, sl_price=None, tp_price=None, sl_pips=None, tp_pips=None, comments=''):

        # ---- handling SL/TP price ----
        if sl_pips is not None:
            if side == 'buy':
                _sl_price = self.data['open'][i] - sl_pips
            else:
                _sl_price = self.data['open'][i] + sl_pips
        else:
            _sl_price = sl_price
                
        if tp_pips is not None:
            if side == 'buy':
                _tp_price = self.data['open'][i] + tp_pips
            else:
                _tp_price = self.data['open'][i] - tp_pips
        else:
            _tp_price = tp_price

        # ---- handling SL/TP order side ----
        if side == 'buy':
            _sl_side = 'sell'
            _tp_side = 'sell'
        
        else:
            _sl_side = 'buy'
            _tp_side = 'buy'

        # ---- market order ----
        self.orders_open.append({'trade_id': self.ORDER_ID,
                                 'entry_time': self.data['datetime'][i], 
                                 'side': side, 
                                 'price': self.data['open'][i], 
                                 'qty': qty, 
                                 'filled': 0,
                                 'type': 'market', 
                                 'comments': comments})
        
        # ---- SL ----
        self.orders_open.append({'trade_id': self.ORDER_ID,
                                 'entry_time': self.data['datetime'][i], 
                                 'side': _sl_side, 
                                 'price': _sl_price, 
                                 'qty': qty, 
                                 'filled': 0,
                                 'type': 'sl', 
                                 'comments': comments})
        
        # ---- TP ----
        self.orders_open.append({'trade_id': self.ORDER_ID,
                                 'entry_time': self.data['datetime'][i], 
                                 'side': _tp_side, 
                                 'price': _tp_price, 
                                 'qty': qty, 
                                 'filled': 0,
                                 'type': 'tp', 
                                 'comments': comments})
        
        self.ORDER_ID += 1
        
        self.TOTAL_ORDERS +=3
        self.OPEN_ORDERS +=3
        self.OPEN_LIMIT_ORDERS +=2
        
        self.TOTAL_TRADES +=1
        self.OPEN_TRADES +=1
    
    @final
    def execute(self):

        for i in range(len(self.df)):

            if self.buy_condition(i):


                self.bracket_order(i,'buy', 1, sl_pips=40, tp_pips=40, comments='BBB')

                pass

            if self.sell_condition(i):

                self.bracket_order(i,'sell', 1, sl_pips=40, tp_pips=40, comments='SSS')
                pass

            self._check_market_sltp(i)

        # recreate pandas from vectors (in case vectors have changed)
        self.df = pd.DataFrame(self.data, index=self.data['datetime'])
        self.tf = pd.DataFrame(self.trades)
        self.oo = pd.DataFrame(self.orders_open)
        self.cc = pd.DataFrame(self.orders_closed)


    # === STOP LOSS AND TAKE PROFIT CHECK FUNCTION ===
    def _check_market_sltp(self, i):

        # open market orders
        order_market = [o for o in self.orders_open if o['type'] == 'market']

        for order in order_market:

            #retreive trade id VALUE
            trade_id = order['trade_id'] 
            order_sl = next(o for o in self.orders_open if o['trade_id']==trade_id and o['type'] =='sl') # return List of sl orders for order id
            order_tp = next(o for o in self.orders_open if o['trade_id']==trade_id and o['type'] =='tp') # return List of tp orders for order id

            #retrieve side
            order_side = order['side']

            ## ====== LONG/SHORT TP CONDITION ======
            if order_side == "buy":
                # buy: candle high >= tp tprice
                tp_condition = self.data['high'][i] >= order_tp['price']
                tp_pnl = order_tp['price'] - order['price']
                sl_pnl = order_sl['price'] - order['price']
            else:
                # sell: candle low <= tp price
                tp_condition = self.data['low'][i] <= order_tp['price']
                tp_pnl = order_tp['price'] - order['price'] *-1
                sl_pnl = order_sl['price'] - order['price'] *-1


            ## ====== TL HIT ======
            if tp_condition: 

                # add trade order record
                
                self.trades.append({    'trade_id': trade_id,
                                        'entry_time': order['entry_time'],
                                        'exit_time': self.data['datetime'][i], 
                                        'entry_price': order['price'], 
                                        'exit_price': order_tp['price'], 
                                        'side': 'sell', 
                                        'qty': 1, 
                                        'filled': 1, 
                                        'sl': order_sl['price'], 
                                        'tp': order_tp['price'], 
                                        'pnl':tp_pnl, 
                                        'comments': ''})
                
                # remove from open_orders
                self.orders_open.remove(order)
                self.orders_open.remove(order_sl)
                self.orders_open.remove(order_tp)

                # add to closed_orders
                order['filled'] = 1
                order_sl['comments'] = 'cancelled'
                order_sl['entry_time'] = self.data['datetime'][i]
                
                order_tp['filled'] = 1
                order_tp['entry_time'] = self.data['datetime'][i]
                
                self.orders_closed.append(order)
                self.orders_closed.append(order_sl)
                self.orders_closed.append(order_tp)

                # update signals and counters
                self.OPEN_LIMIT_ORDERS -=2
                self.OPEN_ORDERS -=1
                self.OPEN_TRADES -=1
                continue
                
            ## ====== LONG/SHORT SL CONDITION ======
            if order_side == "buy":
                # buy: candle low <= sl price
                sl_condition = self.data['low'][i] <= order_sl['price']
            else:
                # sell: candle high >= sl price
                sl_condition = self.data['high'][i] >= order_sl['price']

            ## ====== (BUY) SL HIT ======
            if sl_condition:

                # add trade order record
                pnl = order['price'] - order_sl['price']
                self.trades.append({    'trade_id': trade_id,
                                        'entry_time': order['entry_time'],
                                        'exit_time': self.data['datetime'][i], 
                                        'entry_price': order['price'], 
                                        'exit_price': order_sl['price'], 
                                        'side': 'sell', 
                                        'qty': 1, 
                                        'filled': 1, 
                                        'sl': order_sl['price'], 
                                        'tp': order_tp['price'], 
                                        'pnl':sl_pnl, 
                                        'comments': ''})
                        
                #move remove from open_orders
                self.orders_open.remove(order)
                self.orders_open.remove(order_sl)
                self.orders_open.remove(order_tp)

                # add to closed_orders
                order['filled'] = 1
            
                order_tp['comments'] = 'cancelled'
                order_tp['entry_time'] = self.data['datetime'][i]
                
                order_sl['filled'] = 1
                order_sl['entry_time'] = self.data['datetime'][i]

                self.orders_closed.append(order)
                self.orders_closed.append(order_sl)
                self.orders_closed.append(order_tp)

                # update signals and counters
                self.OPEN_LIMIT_ORDERS -=2
                self.OPEN_ORDERS -=1
                self.OPEN_TRADES -=1
                continue



if __name__ == "__main__":
    pd.set_option("display.max_columns", None)

    # == constants

    SMA_SLOW = 50
    SMA_FAST = 20

    # == load data
    df = load_yfinance("MNQ=F", start="2025-08-16", end="2025-09-16", interval="5m")
    ax, ax2 = fplt.create_plot('MNQ Chart', rows=2)
    fplt.volume_ocv(df[['open', 'close', 'volume']], ax=ax2)
    fplt.candlestick_ochl(df, ax = ax) 

    # == indicators

    #help(talib.SMA)
    df['SMA_slow'] = ta.SMA(df, timeperiod=SMA_SLOW)
    df['SMA_fast'] = ta.SMA(df, timeperiod=SMA_FAST)
    df['crossover'] = pta.cross(df['SMA_fast'], df['SMA_slow'])
    df['crossunder'] = pta.cross(df['SMA_slow'], df['SMA_fast'])


    fplt.plot(df['SMA_slow'] , ax=ax, color="#ff6a00", legend=f"SMA {SMA_SLOW}")
    fplt.plot(df['SMA_fast'] , ax=ax, color="#00ff6a", legend=f"SMA {SMA_FAST}")


    #s.sessions(df, ax=ax)
    vwap(df, ax=ax, mode='daily', color="#0c29cf")
    timeband(df, ax=ax, title="rth", color="#a8a8a830")

    # == strategy logic

    st = Strategy(df)
    st.execute()

    print(st.tf['pnl'].sum())
    
    

    plot_trades(tf=st.tf, cc=st.cc, df=df, ax=ax, boxes=False)

    fplt.show()

