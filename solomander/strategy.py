import finplot as fplt
import talib
import talib.abstract as ta
import pandas_ta as pta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from typing import final

try:
    from .indicators import vwap, timeband, sessions
    from .logger import log, stamp, pront
    from .data import load_yfinance
    from .utils import max_drawdown, sharpe, sortino, timedelta_to_str, print_boxed_title
    from .visuals import plot_trades
except ImportError:
    from indicators import vwap, timeband, sessions
    from logger import log, stamp, pront
    from data import load_yfinance
    from utils import max_drawdown, sharpe, sortino, timedelta_to_str, print_boxed_title
    from visuals import plot_trades
    

class Strategy:
    def __init__(self, df: pd.DataFrame, symbol_info: dict):
        
        # settings 
        self.setting_slippage_entry   = "worst_case"   # off, worst_case, random
        self.setting_slippage_sl      = "off"          # off, worst_case, random
        self.setting_slippage_tp      = "off"          # off, worst_case, random 
        self.setting_rounding_method  = "worst_case"        # nearest, worst_case

        self._setting_rm_buy      = "ceil" if self.setting_rounding_method == "worst_case" else "round"
        self._setting_rm_buy_sl   = "floor" if self.setting_rounding_method == "worst_case" else "round"
        self._setting_rm_buy_tp   = "floor" if self.setting_rounding_method == "worst_case" else "round"
        self._setting_rm_sell     = "floor" if self.setting_rounding_method == "worst_case" else "round"
        self._setting_rm_sell_sl  = "ceil" if self.setting_rounding_method == "worst_case" else "round"
        self._setting_rm_sell_tp  = "ceil" if self.setting_rounding_method == "worst_case" else "round"

        # panda dataframes
        self.df = df                     # main dataframe (candles/indiciators)
        self.tf = pd.DataFrame()         # trade dataframe
        self.oo = pd.DataFrame()         # open orders dataframe
        self.cc = pd.DataFrame()         # closed orders dataframe
        self.df_cum_margin = pd.DataFrame() # cumulative margin over time
        self.df_cum_pnl = pd.DataFrame()    # cumulative pnl sequence series

        # Python lists/ dicts 
        # for dict navigiation, and faster processing (pandas is slow for this)
        self.l_orders_open =[]
        self.l_orders_closed =[]
        self.l_trades =[]
        self.l_cum_margin = []
        self.axs = []                # finplot axes for plotting
        self.data = {}  # will hold numpy arrays of df for faster processing

        # Account info
        self.START_MARGIN = 10000.0      # starting margin
        self.ACTIVE_MARGIN = self.START_MARGIN  # starting margin
    
        # Market info
        self.symbol_data        = symbol_info
        self.MARKET_SYMBOL      = symbol_info.get('symbol', 'Unknown Symbol')
        self.MARKET_NAME        = symbol_info.get('name', 'Unknown Market')
        self.MARKET_TYPE        = symbol_info.get('type', 'futures')           # spot or futures
        self.TICK_CURRENCY      = symbol_info.get('tick_currency', 'USD')      # tick currency
        self.TICK_SIZE          = symbol_info.get('tick_size', 0.25)           # minimum price increment
        self.TICK_PRICE         = symbol_info.get('tick_price', 0.25)          # minimum price increment
        self.POINT_SLIPPAGE     = symbol_info.get('point_slippage', 1.0)
        self.TICK_SPREAD        = symbol_info.get('tick_spread', 0)            # typical spread in ticks
        self.POINT_LEVERAGE     = self.TICK_PRICE/self.TICK_SIZE          # leverage from symbol data

        self.LOT_CURRENCY      = symbol_info.get('lot_currency', 'USD')        # lot currency
        self.LOT_MIN_SIZE      = symbol_info.get('lot_min_size', 1)            # min contract size
        self.LOT_INCREMENT     = symbol_info.get('lot_increment', 1)           # minimum order size increment
        self.FEE_TYPE          = symbol_info.get('fee_type', 'fixed')          # fee round trip per trade
        self.FEE               = symbol_info.get('fee_value', 1.74)  
        
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



    #---- INHERIT AND OVERRIDE THESE METHODS ----
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

    def update_data(self):

        return

    #---- FUNCTIONS -----
    
    def sell_bracket(self, i, qty, sl_price=None, tp_price=None, sl_pips=None, tp_pips=None, comments=''):
        
        # ---- CHECKS ----
        if self._check_qty(qty) == 0:
            return
         # ---- calculate entry, sl and tp prices with slippage and rounding ----
        _entry_price = self._round_to_tick(self.data['open'][i] - self._slippage(self.setting_slippage_entry), self._setting_rm_sell)

        try:
            if tp_pips is not None and sl_pips is not None:
                _sl_price = self._round_to_tick((self.data['open'][i] + sl_pips) + self._slippage(self.setting_slippage_sl), self._setting_rm_sell_sl)
                _tp_price = self._round_to_tick((self.data['open'][i] - tp_pips) + self._slippage(self.setting_slippage_tp), self._setting_rm_sell_tp)
            else:
                _sl_price = self._round_to_tick(sl_price + self._slippage(self.setting_slippage_sl), self._setting_rm_sell_sl)
                _tp_price = self._round_to_tick(tp_price + self._slippage(self.setting_slippage_tp), self._setting_rm_sell_tp)
        except Exception as e:
            log.error(f"Error calculating SL/TP prices: {e}")

          # ---- market order ----
        self.l_orders_open.append({ 'trade_id': self.ORDER_ID,
                                    'entry_time': self.data['datetime'][i], 
                                    'side': 'sell', 
                                    'price': _entry_price, 
                                    'qty': qty, 
                                    'filled': 0,
                                    'type': 'market', 
                                    'comments': comments})
        
        # ---- SL ----
        self.l_orders_open.append({ 'trade_id': self.ORDER_ID,
                                    'entry_time': self.data['datetime'][i], 
                                    'side': "buy", 
                                    'price': _sl_price, 
                                    'qty': qty, 
                                    'filled': 0,
                                    'type': 'sl', 
                                    'comments': comments})
        
        # ---- TP ----
        self.l_orders_open.append({ 'trade_id': self.ORDER_ID,
                                    'entry_time': self.data['datetime'][i], 
                                    'side': "buy", 
                                    'price': _tp_price, 
                                    'qty': qty, 
                                    'filled': 0,
                                    'type': 'tp', 
                                    'comments': comments})
        
        # ---- update counters ----
        self.ORDER_ID += 1
        self.TOTAL_ORDERS +=3
        self.OPEN_ORDERS +=3
        self.OPEN_LIMIT_ORDERS +=2
        self.TOTAL_TRADES +=1
        self.OPEN_TRADES +=1
        self.TOTAL_SHORTS +=1
        
        return

    def buy_bracket(self, i, qty, sl_price=None, tp_price=None, sl_pips=None, tp_pips=None, comments=''):

        # ---- CHECKS ----
        if self._check_qty(qty) == 0:
            return
        # ---- calculate entry, sl and tp prices with slippage and rounding ----
        _entry_price = self._round_to_tick(self.data['open'][i] + self._slippage(self.setting_slippage_entry), self._setting_rm_buy)

        try:
            if tp_pips is not None and sl_pips is not None:
                _sl_price = self._round_to_tick((self.data['open'][i] - sl_pips) - self._slippage(self.setting_slippage_sl), self._setting_rm_buy_sl)
                _tp_price = self._round_to_tick((self.data['open'][i] + tp_pips) - self._slippage(self.setting_slippage_tp), self._setting_rm_buy_tp)
            else:
                _sl_price = self._round_to_tick(sl_price - self._slippage(self.setting_slippage_sl), self._setting_rm_buy_sl)
                _tp_price = self._round_to_tick(tp_price - self._slippage(self.setting_slippage_tp), self._setting_rm_buy_tp)
        except Exception as e:
            log.error(f"Error calculating SL/TP prices: {e}")

          # ---- market order ----
        self.l_orders_open.append({ 'trade_id': self.ORDER_ID,
                                    'entry_time': self.data['datetime'][i], 
                                    'side': 'buy', 
                                    'price': _entry_price, 
                                    'qty': qty, 
                                    'filled': 0,
                                    'type': 'market', 
                                    'comments': comments})
        
        # ---- SL ----
        self.l_orders_open.append({ 'trade_id': self.ORDER_ID,
                                    'entry_time': self.data['datetime'][i], 
                                    'side': "sell", 
                                    'price': _sl_price, 
                                    'qty': qty, 
                                    'filled': 0,
                                    'type': 'sl', 
                                    'comments': comments})
        
        # ---- TP ----
        self.l_orders_open.append({ 'trade_id': self.ORDER_ID,
                                    'entry_time': self.data['datetime'][i], 
                                    'side': "sell", 
                                    'price': _tp_price, 
                                    'qty': qty, 
                                    'filled': 0,
                                    'type': 'tp', 
                                    'comments': comments})
        
        # ---- update counters ----
        self.ORDER_ID += 1
        self.TOTAL_ORDERS +=3
        self.OPEN_ORDERS +=3
        self.OPEN_LIMIT_ORDERS +=2
        self.TOTAL_TRADES +=1
        self.OPEN_TRADES +=1
        self.TOTAL_LONGS +=1
        
        return
    
    def _check_market_sltp(self, i):

        """
        Check for stop loss and take profit conditions for open market orders
        
        """

        # open market orders
        order_market = [o for o in self.l_orders_open if o['type'] == 'market']

        for order in order_market:

            # -------- Retrive order id ---------
            trade_id = order['trade_id'] 
            order_sl = next(o for o in self.l_orders_open if o['trade_id']==trade_id and o['type'] =='sl') # return List of sl orders for order id
            order_tp = next(o for o in self.l_orders_open if o['trade_id']==trade_id and o['type'] =='tp') # return List of tp orders for order id
            order_side = order['side']
            qty = order['qty']

            ## -------- LONG/SHORT TP CONDITION -------
            # = (exit price - entry price) * qty * leverage - fee * qty
            if order_side == "buy":
                tp_condition = self.data['high'][i] >= order_tp['price']
                sl_condition = self.data['low'][i] <= order_sl['price']
                tp_pnl = (order_tp['price'] - order['price']) * qty*self.POINT_LEVERAGE
                sl_pnl = (order_sl['price'] - order['price']) * qty*self.POINT_LEVERAGE
            else:
                tp_condition = self.data['low'][i] <= order_tp['price']
                sl_condition = self.data['high'][i] >= order_sl['price']
                tp_pnl = (order['price'] - order_tp['price']) * qty*self.POINT_LEVERAGE
                sl_pnl = (order['price'] - order_sl['price']) * qty*self.POINT_LEVERAGE

            ## -------- TL HIT --------
            if tp_condition: 
        
                # update signals and counters
                average_price = (order['price'] + order_tp['price'])/2
                fee = self._fee(qty,average_price)
                spread = self._spread(qty)
                
                self.OPEN_LIMIT_ORDERS -=2
                self.OPEN_ORDERS -=1
                self.OPEN_TRADES -=1
                previous_margin = self.ACTIVE_MARGIN
                self.ACTIVE_MARGIN += tp_pnl-fee-spread
                
                if order_side =='buy':
                    self.WINS_LONG +=1
                    self.TOTAL_WINS +=1
                else: 
                    self.WINS_SHORT +=1
                    self.TOTAL_WINS +=1

                # add trade order record
                self.l_trades.append({  'trade_id': trade_id,
                                        'entry_time': order['entry_time'],
                                        'exit_time': self.data['datetime'][i], 
                                        'entry_price': order['price'], 
                                        'exit_price': order_tp['price'], 
                                        'side': order_side, 
                                        'qty': qty, 
                                        'filled': qty, 
                                        'sl': order_sl['price'], 
                                        'tp': order_tp['price'], 
                                        'raw_pnl': tp_pnl, 
                                        'fee': fee, 
                                        'spread': spread, 
                                        'pnl':tp_pnl - fee - spread,
                                        'margin': self.ACTIVE_MARGIN,
                                        'return': (tp_pnl - fee-spread)/(previous_margin) if previous_margin !=0 else 0,
                                        'comments': ''})
                
                # remove from open_orders
                self.l_orders_open.remove(order)
                self.l_orders_open.remove(order_sl)
                self.l_orders_open.remove(order_tp)

                # add to closed_orders
                order['filled'] = 1
                order_sl['comments'] = 'cancelled'
                order_sl['entry_time'] = self.data['datetime'][i]
                
                order_tp['filled'] = 1
                order_tp['entry_time'] = self.data['datetime'][i]
                
                self.l_orders_closed.append(order)
                self.l_orders_closed.append(order_sl)
                self.l_orders_closed.append(order_tp)
                continue
                

            # -------- SL HIT --------
            if sl_condition:

                # update signals and counters
                average_price = (order['price'] + order_sl['price'])/2
                fee = self._fee(qty,average_price)
                spread = self._spread(qty)

                self.OPEN_LIMIT_ORDERS -=2
                self.OPEN_ORDERS -=1
                self.OPEN_TRADES -=1
                previous_margin = self.ACTIVE_MARGIN
                self.ACTIVE_MARGIN += sl_pnl-fee-spread


                # add trade order record
                self.l_trades.append({  'trade_id': trade_id,
                                        'entry_time': order['entry_time'],
                                        'exit_time': self.data['datetime'][i], 
                                        'entry_price': order['price'], 
                                        'exit_price': order_sl['price'], 
                                        'side': order_side, 
                                        'qty': qty, 
                                        'filled': qty, 
                                        'sl': order_sl['price'], 
                                        'tp': order_tp['price'], 
                                        'raw_pnl': sl_pnl,
                                        'fee':fee,
                                        'spread': spread,
                                        'pnl':sl_pnl-fee-spread,
                                        'margin': self.ACTIVE_MARGIN,
                                        'return': (sl_pnl-fee-spread)/(previous_margin) if previous_margin !=0 else 0,
                                        'comments': ''})
                        
                #move remove from open_orders
                self.l_orders_open.remove(order)
                self.l_orders_open.remove(order_sl)
                self.l_orders_open.remove(order_tp)

                # add to closed_orders
                order['filled'] = 1
            
                order_tp['comments'] = 'cancelled'
                order_tp['entry_time'] = self.data['datetime'][i]
                
                order_sl['filled'] = 1
                order_sl['entry_time'] = self.data['datetime'][i]

                self.l_orders_closed.append(order)
                self.l_orders_closed.append(order_sl)
                self.l_orders_closed.append(order_tp)
                continue

    #---- INTERNAL FUNCTIONS -----
    @final
    def _fee(self, qty, price):

        if self.FEE_TYPE == "percent":
            # notional value (=price * qty) x fee rate 
            return price*qty*self.FEE*2
        else:
            return self.FEE * qty*2

    @final
    def _spread(self, qty):

        return self.TICK_SPREAD * self.TICK_PRICE * qty

    @final
    def _round_to_tick(self, price, method):

        if method == "round":
            return round(price / self.TICK_SIZE) * self.TICK_SIZE
        elif method == "floor":
            return (price // self.TICK_SIZE) * self.TICK_SIZE
        elif method == "ceil":
            return (-( -price // self.TICK_SIZE)) * self.TICK_SIZE
        else:
            raise ValueError(f"Unknown rounding method: {method}")

    @final
    def _slippage(self, setting):

        if setting == "off":
            return 0.0
        elif setting == "worst_case":
            return self.POINT_SLIPPAGE
        elif setting == "random":
            return self._round_to_tick(np.random.uniform(-self.POINT_SLIPPAGE, self.POINT_SLIPPAGE), "round")

    @final
    def _check_qty(self, qty):

        if qty < self.LOT_MIN_SIZE:
            log.warning(f"Order qty {qty} is less than min lot size {self.LOT_MIN_SIZE}")  
            return 0
        if not math.isclose(qty / self.LOT_INCREMENT % 1, 0, abs_tol=9e-03):
            log.warning(f"Order qty {qty} not compatible with lot increment {self.LOT_INCREMENT}")
            return 0
        return 1

    
    






if __name__ == "__main__":

    pass


    
    



