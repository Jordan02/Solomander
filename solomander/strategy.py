import finplot as fplt
import talib
import talib.abstract as ta
import pandas_ta as pta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import final

try:
    from .indicators import vwap, timeband, sessions
    from .logger import log, stamp, pront
    from .data import load_yfinance
    from .analysis import max_drawdown, sharpe, sortino
    from .visuals import plot_trades
except ImportError:
    from indicators import vwap, timeband, sessions
    from logger import log, stamp, pront
    from data import load_yfinance
    from analysis import max_drawdown, sharpe, sortino
    from visuals import plot_trades
    

class Strategy:
    def __init__(self, df : pd.DataFrame, **kwargs):
        
        # panda dataframes
        self.df = df                     # main dataframe (candles/indiciators)
        self.tf = pd.DataFrame()         # trade dataframe
        self.oo = pd.DataFrame()         # open orders dataframe
        self.cc = pd.DataFrame()         # closed orders dataframe
        self.CUM_MARGIN = pd.DataFrame() # cumulative margin over time
        self.CUM_PNL = pd.DataFrame()    # cumulative pnl sequence series

        # Python lists 
        # for dict navigiation, and faster processing (pandas is slow for this)
        self.orders_open =[]
        self.orders_closed =[]
        self.trades =[]
        self.cum_margin = []
        self.axs = []                # finplot axes for plotting

        # python dicts
        self.data = {}  # will hold numpy arrays of df for faster processing

        # Account info
        self.STARTING_MARGIN = 10000.0      # starting margin
        self.MARGIN = self.STARTING_MARGIN  # starting margin

        # Market info
        self.LEVERAGE = 1       # leverage
        self.FEE = 1.74         # fee round trip per trade
        self.SLIPPAGE = 0.0     # slippage per trade
        self.TEST_DAYS = np.busday_count(df.index[0].date(), df.index[-1].date()) # number of business days in test period
        self.TEST_DAYS = max(1, self.TEST_DAYS) 

        # ==== DYNANIMC signals, Counters and metrics (will be updated during execution) ====
        
        self.ORDER_ID = 0
        self.TOTAL_ORDERS = 0
        self.OPEN_ORDERS = 0
        self.OPEN_LIMIT_ORDERS = 0
        self.TOTAL_TRADES = 0
        self.OPEN_TRADES = 0
        
        self.TOTAL_SHORTS = 0 #check
        self.TOTAL_LONGS = 0 #check
        self.WINS_SHORT = 0 #check
        self.WINS_LONG = 0 #check
        self.TOTAL_WINS = 0 #check
        
        # ==== STATIC Performance metrics (to be calculated at end of execution) ====
        
        self.WIN_RATE = 0.0 #check
        self.WIN_RATE_LONG = 0.0 
        self.WIN_RATE_SHORT = 0.0 
        self.AVG_PROFIT = 0.0 #check
        self.AVG_LOSS = 0.0 #check
        self.TOTAL_PROFIT = 0.0 #check
        self.TOTAL_LOSS = 0.0 #check

        self.PAYOFF_RATIO = 0.0 #check
        self.PROFIT_FACTOR = 0.0 #check
        self.SHARPE_RATIO_ANNUAL = 0.0 #check
        self.SORTINO_RATIO_ANNUAL = 0.0 #check
        self.SHARPE_RATIO_DAILY = 0.0 #check
        self.SORTINO_RATIO_DAILY = 0.0 #check
        self.MAX_DRAWDOWN = 0.0 #check
        self.PNL = 0.0 #check
        self.PNL_MDD_RATIO = 0.0 #check

        # ==== All Kwargs are stored as Strategy params ====
        self.INPUT_PARAMS = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def buy_condition(self, i):
        
        # default strategy: crossover of 9 and 21 EMA
        # inherit and override this method for custom strategy
        return 0
        
    def sell_condition(self, i):
        
        # default strategy: crossover of 9 and 21 EMA
        # inherit and override this method for custom strategy
        #return self.data['crossunder'][i] > 0 and self.TOTAL_TRADES < 2
        return 0

    def buy_action(self, i):

        return 0

    def sell_action(self, i):
        
        return

    def plots(self, rows=2):
        
        if rows < 2:
            stamp.critical("Strategy.plots(): rows must be >=2")
            return
        
        self.axs = fplt.create_plot('MNQ Chart', rows=rows)

        # standard candles
        fplt.volume_ocv(self.df[['open', 'close', 'volume']], ax=self.axs[0].overlay())
        fplt.candlestick_ochl(self.df, ax=self.axs[0])
        plot_trades(tf=self.tf, cc=self.cc, df=self.df, ax=self.axs[0], boxes=True)
        
        # PnL chart
        fplt.add_line((self.df.index[0], self.STARTING_MARGIN), (self.df.index[-1], self.STARTING_MARGIN), ax=self.axs[-1], color="#130000", style="--")
        fplt.plot(self.CUM_MARGIN, ax=self.axs[-1], color="#ff6a00", legend="cumulative Pnl")


    @final
    def show(self):
        self.plots()
        fplt.show()
        return




    @final
    def print_metrics(self):

        label_width = 18  # adjust so colons line up
        
        print("\n=== TEST METRICS ===")
        print(f"{'Start:':<{label_width}} {self.df.index[0]}")
        print(f"{'End:':<{label_width}} {self.df.index[-1]}")
        print(f"{'Days:':<{label_width}} {self.TEST_DAYS}")
        print(f"{'Total Trades:':<{label_width}} {self.TOTAL_TRADES}")
        print(f"{'Total Longs:':<{label_width}} {self.TOTAL_LONGS}")
        print(f"{'Total Shorts:':<{label_width}} {self.TOTAL_SHORTS}")

        print("\n=== STRATEGY METRICS ===")
        print(f"{'PNL:':<{label_width}} ${self.PNL:.2f}")
        print(f"{'Win Rate Long:':<{label_width}} {self.WIN_RATE_LONG:.2f}%")
        print(f"{'Win Rate Short:':<{label_width}} {self.WIN_RATE_SHORT:.2f}%")
        print(f"{'Win Rate Total:':<{label_width}} {self.WIN_RATE:.2f}%")
        print(f"{'Average Profit:':<{label_width}} ${self.AVG_PROFIT:.2f}")
        print(f"{'Average Loss:':<{label_width}} ${self.AVG_LOSS:.2f}")
        print(f"{'Max Drawdown:':<{label_width}} ${self.MAX_DRAWDOWN:.2f}")
        print(f"{'PnL/MDD Ratio:':<{label_width}} {self.PNL_MDD_RATIO:.2f}")
        print(f"{'Payoff Ratio:':<{label_width}} {self.PAYOFF_RATIO:.2f}")
        print(f"{'Profit Factor:':<{label_width}} {self.PROFIT_FACTOR:.2f}")
        print(f"{'Sharpe Ratio(Y):':<{label_width}} {self.SHARPE_RATIO_ANNUAL:.2f}")
        print(f"{'Sharpe Ratio(D):':<{label_width}} {self.SHARPE_RATIO_DAILY:.2f}")
        print(f"{'Sortino Ratio(Y):':<{label_width}} {self.SORTINO_RATIO_ANNUAL:.2f}")
        print(f"{'Sortino Ratio(D):':<{label_width}} {self.SORTINO_RATIO_DAILY:.2f}")
        return

    @final
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

        if side == 'buy':
            self.TOTAL_LONGS +=1
        else:
            self.TOTAL_SHORTS +=1
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

        if side == 'buy':
            self.TOTAL_LONGS +=1
        else:
            self.TOTAL_SHORTS +=1
        return
    
    @final
    def execute(self):

        # convert df to numpy arrays for faster processing.
        self.data = {col: self.df[col].to_numpy().copy() for col in self.df.columns}
        self.data['datetime'] = self.df.index.to_numpy().copy() # Copy allows overriding of values

        # Fixed Strategy loop:
        for i in range(len(self.df)):

            if self.buy_condition(i):
                
                self.buy_action(i)
                pass

            if self.sell_condition(i):

                self.sell_action(i)
                pass

            self._check_market_sltp(i)
            self.cum_margin.append(self.MARGIN)  # track margin over time

        # recreate pandas from vectors (in case vectors have changed)
        self.df = pd.DataFrame(self.data, index=self.data['datetime'])
        self.tf = pd.DataFrame(self.trades)
        self.oo = pd.DataFrame(self.orders_open)
        self.cc = pd.DataFrame(self.orders_closed)
        self.CUM_MARGIN = pd.DataFrame({'datetime': self.data['datetime'], 'margin': self.cum_margin}).set_index('datetime')
        self.CUM_PNL = self.tf[['pnl']].cumsum().rename(columns={'pnl':'cum_pnl'}).shift(1).fillna(0)
        
        # calculate performance metrics
        profitable_trades = self.tf.loc[self.tf['pnl']>0]
        losing_trades = self.tf.loc[self.tf['pnl']<0]

        self.PNL = self.tf['pnl'].sum() if not self.tf['pnl'].empty else 0
        self.WIN_RATE_LONG = (self.WINS_LONG / self.TOTAL_LONGS * 100) if self.TOTAL_LONGS >0 else 0
        self.WIN_RATE_SHORT = (self.WINS_SHORT / self.TOTAL_SHORTS * 100) if self.TOTAL_SHORTS >0 else 0
        self.WIN_RATE = (self.TOTAL_WINS / self.TOTAL_TRADES * 100) if self.TOTAL_TRADES >0 else 0
        self.AVG_PROFIT = profitable_trades['pnl'].mean() if not profitable_trades['pnl'].empty else 0
        self.AVG_LOSS = losing_trades['pnl'].mean() if not losing_trades['pnl'].empty else 0
        self.TOTAL_PROFIT = profitable_trades['pnl'].sum() if not profitable_trades['pnl'].empty else 0
        self.TOTAL_LOSS = losing_trades['pnl'].sum() if not losing_trades['pnl'].empty else 0
        self.PAYOFF_RATIO = (self.AVG_PROFIT / abs(self.AVG_LOSS)) if self.AVG_LOSS !=0 else 0
        self.MAX_DRAWDOWN = max_drawdown(self.tf['pnl']) if not self.tf['pnl'].empty else 0
        self.PNL_MDD_RATIO = (self.PNL / abs(self.MAX_DRAWDOWN)) if self.MAX_DRAWDOWN !=0 else 0

        self.PROFIT_FACTOR = (self.TOTAL_PROFIT / (self.TOTAL_LOSS*-1)) if self.TOTAL_LOSS !=0 else 0

        # annualized ratios
        self.SHARPE_RATIO_DAILY = sharpe(self.tf, type="daily")
        self.SHARPE_RATIO_ANNUAL = self.SHARPE_RATIO_DAILY * np.sqrt(252)

        self.SORTINO_RATIO_DAILY = sortino(self.tf, type="daily")
        self.SORTINO_RATIO_ANNUAL = self.SORTINO_RATIO_DAILY * np.sqrt(252) 

    @final
    def _check_market_sltp(self, i):

        """
        Check for stop loss and take profit conditions for open market orders
        
        """

        # open market orders
        order_market = [o for o in self.orders_open if o['type'] == 'market']

        for order in order_market:

            #retreive trade id VALUE
            trade_id = order['trade_id'] 
            order_sl = next(o for o in self.orders_open if o['trade_id']==trade_id and o['type'] =='sl') # return List of sl orders for order id
            order_tp = next(o for o in self.orders_open if o['trade_id']==trade_id and o['type'] =='tp') # return List of tp orders for order id

            #retrieve side
            order_side = order['side']
            qty = order['qty']

            ## ====== LONG/SHORT TP CONDITION ======
            # = (exit price - entry price) * qty * leverage - fee * qty
            if order_side == "buy":
                # buy: candle high >= tp tprice
                tp_condition = self.data['high'][i] >= order_tp['price']
                tp_pnl = (order_tp['price'] - order['price']) * qty*self.LEVERAGE - (self.FEE * qty)
                sl_pnl = (order_sl['price'] - order['price']) * qty*self.LEVERAGE - (self.FEE * qty)
            else:
                # sell: candle low <= tp price
                tp_condition = self.data['low'][i] <= order_tp['price']
                tp_pnl = (order['price'] - order_tp['price']) * qty*self.LEVERAGE - (self.FEE * qty)
                sl_pnl = (order['price'] - order_sl['price']) * qty*self.LEVERAGE - (self.FEE * qty)

            ## ====== TL HIT ======
            if tp_condition: 
            
                # update signals and counters
                self.OPEN_LIMIT_ORDERS -=2
                self.OPEN_ORDERS -=1
                self.OPEN_TRADES -=1
                previous_margin = self.MARGIN
                self.MARGIN += tp_pnl
                
                if order_side =='buy':
                    self.WINS_LONG +=1
                    self.TOTAL_WINS +=1
                else: 
                    self.WINS_SHORT +=1
                    self.TOTAL_WINS +=1

                # add trade order record
                
                self.trades.append({    'trade_id': trade_id,
                                        'entry_time': order['entry_time'],
                                        'exit_time': self.data['datetime'][i], 
                                        'entry_price': order['price'], 
                                        'exit_price': order_tp['price'], 
                                        'side': 'sell', 
                                        'qty': qty, 
                                        'filled': qty, 
                                        'sl': order_sl['price'], 
                                        'tp': order_tp['price'], 
                                        'pnl':tp_pnl,
                                        'margin': self.MARGIN,
                                        'return': tp_pnl/(previous_margin) if previous_margin !=0 else 0,
                                        'fee':self.FEE * qty, 
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

                # update signals and counters
                self.OPEN_LIMIT_ORDERS -=2
                self.OPEN_ORDERS -=1
                self.OPEN_TRADES -=1
                previous_margin = self.MARGIN
                self.MARGIN += sl_pnl

                if order_side =='buy':
                    pass
                else: 
                    pass

                # add trade order record
                self.trades.append({    'trade_id': trade_id,
                                        'entry_time': order['entry_time'],
                                        'exit_time': self.data['datetime'][i], 
                                        'entry_price': order['price'], 
                                        'exit_price': order_sl['price'], 
                                        'side': 'sell', 
                                        'qty': qty, 
                                        'filled': qty, 
                                        'sl': order_sl['price'], 
                                        'tp': order_tp['price'], 
                                        'pnl':sl_pnl,
                                        'margin': self.MARGIN,
                                        'return': sl_pnl/(previous_margin) if previous_margin !=0 else 0,
                                        'fee':self.FEE * qty,
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
                continue



if __name__ == "__main__":

    pd.set_option("display.max_columns", None)

    # ==== CONSTANTS =====

    SMA_SLOW = 50
    SMA_FAST = 20
    RR = 2

    # ==== DATA AND INDICATORS =====
    df = load_yfinance("MNQ=F", start="2025-08-16", end="2025-09-16", interval="5m")

    #help(talib.SMA)
    df['SMA_slow'] = ta.SMA(df, timeperiod=SMA_SLOW)
    df['SMA_fast'] = ta.SMA(df, timeperiod=SMA_FAST)
    df['crossover'] = pta.cross(df['SMA_fast'], df['SMA_slow'])
    df['crossunder'] = pta.cross(df['SMA_slow'], df['SMA_fast'])
    df['NY'] = sessions(df)['NY']
    df['vwap'] = vwap(df, mode="daily")

    # ==== STRATEGY EXECUTION =====


    class strat1(Strategy):

        # ===== BUY LOGIC =====
        def buy_condition(self, i):
            
            return self.data['crossover'][i] > 0 and self.OPEN_TRADES < 1

        
        def buy_action(self, i):
            
            pre_i = max(0, i-1)
            rr = RR
            sl = self.data['high'][pre_i] - self.data['low'][pre_i]
            tp = sl * rr

            self.bracket_order(i,'buy', 10, sl_pips=sl, tp_pips=tp, comments='BBB')
            
            return 
        
        # ===== SELL LOGIC =====
        def sell_condition(self, i):
            
            return self.data['crossunder'][i] > 0 and self.OPEN_TRADES < 1
        
        def sell_action(self, i):
        
            pre_i = max(0, i-1)
            rr = RR
            sl = self.data['high'][pre_i] - self.data['low'][pre_i]
            tp = sl * rr

            self.bracket_order(i,'sell', 10, sl_pips=sl, tp_pips=tp, comments='SSS')
            
            return 
    
    
    st = strat1(df)
    st.FEE = 1.74
    st.LEVERAGE = 2

    st.execute()


    # ==== PLOTTING VISUALS =====

    
    ax, ax2 = fplt.create_plot('MNQ Chart', rows=2)
    fplt.volume_ocv(df[['open', 'close', 'volume']], ax=ax.overlay())
    fplt.candlestick_ochl(df, ax = ax) 
    #fplt.set_y_range(1.0, 1.2, ax=ax2)
    #fplt.set_y_scale('log', ax=ax2)

    pront.info(st.tf.head(20))

    fplt.add_line((df.index[0], st.STARTING_MARGIN), (df.index[-1], st.STARTING_MARGIN), ax = ax2, color="#130000", style="--")
    fplt.plot(st.CUM_MARGIN, ax=ax2, color="#ff6a00", legend="cumulative Pnl")

    pront.info(st.CUM_PNL.head(20))

    fplt.plot(df['SMA_slow'] , ax=ax, color="#ff6a00", legend=f"SMA {SMA_SLOW}")
    fplt.plot(df['SMA_fast'] , ax=ax, color="#00ff6a", legend=f"SMA {SMA_FAST}")
    fplt.plot(df['vwap'], ax=ax, color="#219bec", legend="VWAP")


    

    # ==== OUTPUTS =====

    cum_pnl = st.CUM_PNL['cum_pnl']


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



    fplt.show()
    plt.show()


    print(st.tf['pnl'].sum())


    
    



