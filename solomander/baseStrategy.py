from enum import Enum
import finplot as fplt
import talib
import talib.abstract as ta
import pandas_ta as pta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from enum import Enum, auto
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from typing import final

try:
    from .indicators import vwap, timeband, sessions
    from .logger import log, stamp, pront
    from .data import load_yfinance
    from .utils import max_drawdown, sharpe, sortino, timedelta_to_str, print_boxed_title
    from .visuals import plot_trades, basic_graph
except ImportError:
    from indicators import vwap, timeband, sessions
    from logger import log, stamp, pront
    from data import load_yfinance
    from utils import max_drawdown, sharpe, sortino, timedelta_to_str, print_boxed_title
    from visuals import plot_trades, basic_graph
    


class Setting(Enum):
    CEIL = auto()
    FLOOR = auto()
    ROUND = auto()

    SLIP_OFF = auto()
    SLIP_WORST_CASE = auto()
    SLIP_RANDOM = auto()

    ROUND_NEAREST = auto()
    ROUND_WORST_CASE = auto()

    MODE_BACKTEST = auto()
    MODE_LIVE = auto()
    MODE_TEST = auto()
    MODE_NONE = auto()  
    

class Strategy:

    def __init__(self, **kwargs):
        
        # ---- settings ----
        self.setting_slippage_entry    = Setting.SLIP_OFF
        self.setting_slippage_sl       = Setting.SLIP_OFF
        self.setting_slippage_tp       = Setting.SLIP_OFF
        self.setting_rounding_method   = Setting.ROUND_WORST_CASE
        self.setting_track_all_metrics = False
        self.setting_listen_time       = None

        # ---- panda dataframes ----
        self.df = None                      # main dataframe (candles/indiciators)
        self.tf = None                      # trade dataframe
        self.oo = None                      # open orders dataframe
        self.cc = None                      # closed orders dataframe
        self.df_cum_margin = None         # cumulative margin dataframe for plotting
        self.df_cum_pnl = None         # cumulative margin dataframe for plotting

        # ---- Order and data lists ----
        self.l_orders_open =[]
        self.l_orders_closed =[]
        self.l_trades =[]
        self.axs = []                # finplot axes for plotting
        self.data = {}               # will hold numpy arrays of df for faster processing
        
        # ---- Market info (static) ----
        self.symbol_data = {} 
        self.START_MARGIN = 10000.0     # starting margin (KWARGS UPDATED)
        self.MARKET_SYMBOL = None   
        self.MARKET_NAME = None       
        self.MARKET_TYPE = None           # spot or futures
        self.TICK_CURRENCY = None         # tick currency
        self.TICK_SIZE = None            # minimum price increment
        self.TICK_PRICE = None           # minimum price increment
        self.POINT_SLIPPAGE = None
        self.TICK_SPREAD = None          # typical spread in ticks
        self.LOT_CURRENCY = None          # lot currency
        self.LOT_MIN_SIZE = None        # min contract size
        self.LOT_INCREMENT = None        # minimum order size increment
        self.FEE_TYPE = None              # fee round trip per trade
        self.FEE = None                  # fee amount

        # ---- order metrics (dynamic) ---- update when trades are placed/closed
        
        self.ORDER_ID = 0
        self.MARGIN = 0 # (KWARGS UPDATED)
        self.WINS_SHORT = 0 
        self.WINS_LONG = 0 
        
        self.LAST_PNL = 0.0 
        self.LAST_RAW_PNL = 0.0
        self.LAST_CUMSUM_PNL = 0.0
        self.LAST_RETURN = 0.0

        self.TOTAL_ORDERS = 0
        self.TOTAL_TRADES = 0
        self.TOTAL_SHORTS = 0 
        self.TOTAL_LONGS = 0 
        self.TOTAL_WINS = 0 

        self.OPEN_ORDERS = 0
        self.OPEN_LIMIT_ORDERS = 0
        self.OPEN_TRADES = 0

        # ---- metrics (historical track) ---- included within self.update_metrics() PER CLOSED TRADE
        
        self.l_MARGIN = []
        self.l_PNL = []
        self.l_CUMSUM_PNL = []
        self.l_RAW_PNL = []
        self.l_RETURN = []
        self.l_DATETIME_RETURN = [] # updated in _check_sltp() for returns
        self.l_PROFIT = []
        self.l_LOSS = []
        
        self.l_DATETIME_PNL = []     # time vs pnl for plotting
        self.l_DATETIME_MARGIN = []  # time vs margin for plotting
        self.l_DATETIMES = []        # time vs dates for plotting

        if self.setting_track_all_metrics:
            self.l_PAYOFF_RATIO = []
            self.l_PROFIT_FACTOR = []
            self.l_SHARPE_RATIO_ANNUAL = []
            self.l_SHARPE_RATIO_DAILY = []
            self.l_SORTINO_RATIO_ANNUAL = []
            self.l_SORTINO_RATIO_DAILY = []
            self.l_MAX_DRAWDOWN = []
            self.l_PNL_MDD_RATIO = []
            self.l_WIN_RATE = []
            self.l_WIN_RATE_SHORT = []
            self.l_WIN_RATE_LONG = []

        
        # ---- metrics (active/dynamic/snapshot/latest) ---- update after trade closes, included within self.update_metrics()
        
        self.TOTAL_MARGIN = 0.0
        self.TOTAL_PNL = 0.0
        self.TOTAL_RAW_PNL = 0.0
        self.TOTAL_RETURN = 0.0
        self.TOTAL_PROFITS = 0.0
        self.TOTAL_LOSSES = 0.0 

        self.TOTAL_PAYOFF_RATIO = 0.0
        self.TOTAL_PROFIT_FACTOR = 0.0
        self.TOTAL_SHARPE_RATIO_ANNUAL = 0.0
        self.TOTAL_SHARPE_RATIO_DAILY = 0.0
        self.TOTAL_SORTINO_RATIO_ANNUAL = 0.0
        self.TOTAL_SORTINO_RATIO_DAILY = 0.0
        self.TOTAL_MAX_DRAWDOWN = 0.0
        self.TOTAL_PNL_MDD_RATIO = 0.0
        self.TOTAL_WIN_RATE = 0.0
        self.TOTAL_WIN_RATE_LONG = 0.0
        self.TOTAL_WIN_RATE_SHORT = 0.0

        self.AVERAGE_PROFIT = 0.0
        self.AVERAGE_LOSS = 0.0
        self.AVERAGE_RETURN = 0.0

        # ----- All **kwargs stored as params -----
        self.INPUT_PARAMS = kwargs
        self.INITIAL_KWARGS = kwargs.copy() # store original kwargs for reference
        for key, value in kwargs.items():
            setattr(self, key, value)

        # ----- params for recalculation -----                                                            
        self.MARGIN             = self.START_MARGIN                                                           # starting margin
        self.POINT_LEVERAGE     = self.TICK_PRICE/self.TICK_SIZE if self.TICK_SIZE is not None else 0         # leverage from symbol data
        self.TICK_SLIPPAGE      = self.POINT_SLIPPAGE/ self.TICK_SIZE if self.TICK_SIZE is not None else 0    # slippage in price terms

        self._setting_rm_buy      = Setting.CEIL if self.setting_rounding_method == Setting.ROUND_WORST_CASE else Setting.ROUND
        self._setting_rm_buy_sl   = Setting.FLOOR if self.setting_rounding_method == Setting.ROUND_WORST_CASE else Setting.ROUND
        self._setting_rm_buy_tp   = Setting.FLOOR if self.setting_rounding_method == Setting.ROUND_WORST_CASE else Setting.ROUND
        self._setting_rm_sell     = Setting.FLOOR if self.setting_rounding_method == Setting.ROUND_WORST_CASE else Setting.ROUND
        self._setting_rm_sell_sl  = Setting.CEIL if self.setting_rounding_method == Setting.ROUND_WORST_CASE else Setting.ROUND
        self._setting_rm_sell_tp  = Setting.CEIL if self.setting_rounding_method == Setting.ROUND_WORST_CASE else Setting.ROUND

        # ----- Other parameters for runners -----
        self.TEST_MODE = Setting.MODE_NONE
        
        self.TIME_INTERVAL = None
        self.TIME_INTERVAL_STR = "Unknown"
        self.TIME_START = None
        self.TIME_END = None
        self.TIME_ELAPSED = None
        self.TIME_WORK_DAYS = None

        self.POLL_INTERVAL = None
        self.CANDLE_BUFFER = None
        self.DEVIATION = None

        self.STRATEGY_NAME = self.__class__.__name__

    # ====== INHERIT AND OVERRIDE THESE METHODS ======
    def buy_condition(self, i):
        """ Condition to execute buy_action, return True/False """
        return 0
        
    def sell_condition(self, i):
        """ Condition to execute sell_action, return True/False """
        return 0

    def buy_action(self, i):
        """ Logic to execute when buy_condition is true """
        return 0

    def sell_action(self, i):
        """ Logic to execute when sell_condition is true """
        return

    def update_data(self):
        """ add needed df updates here, e.g. indicators """
        return
    
    def loop_update(self, i):
        """ updates to be made each loop (data received) """

        self.l_DATETIME_PNL.append(self.LAST_PNL)
        self.l_DATETIME_MARGIN.append(self.MARGIN)
        self.l_DATETIMES.append(self.df.index[i])

        return

    # ====== FUNCTIONS ======
    
    def _update_data_arrays(self):
        self.update_data()
        self.data = {col: self.df[col].to_numpy().copy() for col in self.df.columns}
        self.data['datetime'] = self.df.index.to_numpy().copy() # Copy allows overriding of values
        return

    def update_metrics(self):
       
        # ---- mandatory list updates first ----
        self.l_MARGIN.append(self.MARGIN)
        self.l_PNL.append(self.LAST_PNL)
        self.l_CUMSUM_PNL.append(self.LAST_CUMSUM_PNL)

        self.l_RAW_PNL.append(self.LAST_RAW_PNL)
        self.l_RETURN.append(self.LAST_RETURN)

        if self.LAST_PNL > 0:
            self.l_PROFIT.append(self.LAST_PNL)
        else:
            self.l_LOSS.append(self.LAST_PNL)

        # ---- then calculate metric placeholders ----

        AVG_PROFIT = sum(self.l_PROFIT) / len(self.l_PROFIT) if len(self.l_PROFIT) >0 else 0
        AVG_LOSS = sum(self.l_LOSS) / len(self.l_LOSS) if len(self.l_LOSS) >0 else 0
        TOTAL_PROFIT = sum(self.l_PROFIT) 
        TOTAL_LOSS = sum(self.l_LOSS) 

        PAYOFF_RATIO = AVG_PROFIT / abs(AVG_LOSS) if AVG_LOSS != 0 else 0
        PROFIT_FACTOR = (TOTAL_PROFIT / (TOTAL_LOSS*-1)) if TOTAL_LOSS != 0 else 0

        if len(self.l_RETURN) > 1:
            SHARPE_RATIO_DAILY = sharpe(self.l_RETURN, self.l_DATETIME_RETURN, mode="daily")
            SORTINO_RATIO_DAILY = sortino(self.l_RETURN, self.l_DATETIME_RETURN, mode="daily")
        else:
            SHARPE_RATIO_DAILY = 0
            SORTINO_RATIO_DAILY = 0

        SHARPE_RATIO_ANNUAL = SHARPE_RATIO_DAILY * np.sqrt(256)
        SORTINO_RATIO_ANNUAL = SORTINO_RATIO_DAILY * np.sqrt(256)
        MAX_DRAWDOWN = max_drawdown(self.l_PNL) 
        PNL_MDD_RATIO = (self.TOTAL_PNL_MDD_RATIO / abs(MAX_DRAWDOWN)) if MAX_DRAWDOWN != 0 else 0
        WIN_RATE = (self.TOTAL_WINS / self.TOTAL_TRADES) if self.TOTAL_TRADES >0 else 0
        WIN_RATE_SHORT = (self.WINS_SHORT / self.TOTAL_SHORTS) if self.TOTAL_SHORTS >0 else 0
        WIN_RATE_LONG = (self.WINS_LONG / self.TOTAL_LONGS) if self.TOTAL_LONGS >0 else 0

        # ---- update total metrics ----

        self.TOTAL_MARGIN = self.MARGIN
        self.TOTAL_PNL += self.LAST_PNL
        self.TOTAL_RAW_PNL += self.LAST_RAW_PNL
        self.TOTAL_RETURN = ((self.TOTAL_MARGIN - self.START_MARGIN) / self.START_MARGIN) if self.START_MARGIN !=0 else 0
        self.TOTAL_PROFITS = sum(self.l_PROFIT)
        self.TOTAL_LOSSES = sum(self.l_LOSS)

        self.TOTAL_PAYOFF_RATIO = PAYOFF_RATIO
        self.TOTAL_PROFIT_FACTOR = PROFIT_FACTOR
        self.TOTAL_SHARPE_RATIO_DAILY = SHARPE_RATIO_DAILY
        self.TOTAL_SHARPE_RATIO_ANNUAL = SHARPE_RATIO_ANNUAL
        self.TOTAL_SORTINO_RATIO_DAILY = SORTINO_RATIO_DAILY
        self.TOTAL_SORTINO_RATIO_ANNUAL = SORTINO_RATIO_ANNUAL
        self.TOTAL_MAX_DRAWDOWN = MAX_DRAWDOWN
        self.TOTAL_PNL_MDD_RATIO = self.TOTAL_PNL/self.TOTAL_MAX_DRAWDOWN*-1 if self.TOTAL_MAX_DRAWDOWN !=0 else 0
        self.TOTAL_WIN_RATE = WIN_RATE
        self.TOTAL_WIN_RATE_LONG = WIN_RATE_LONG
        self.TOTAL_WIN_RATE_SHORT = WIN_RATE_SHORT

        self.AVERAGE_PROFIT = sum(self.l_PROFIT)/len(self.l_PROFIT) if len(self.l_PROFIT) >0 else 0
        self.AVERAGE_LOSS = sum(self.l_LOSS)/len(self.l_LOSS) if len(self.l_LOSS) >0 else 0
        self.AVERAGE_RETURN = sum(self.l_RETURN)/len(self.l_RETURN) if len(self.l_RETURN) >0 else 0

         # ---- Additionally track metrics ----
        if self.setting_track_all_metrics:
            self.l_PAYOFF_RATIO.append(PAYOFF_RATIO)
            self.l_PROFIT_FACTOR.append(PROFIT_FACTOR)
            self.l_SHARPE_RATIO_ANNUAL.append(SHARPE_RATIO_ANNUAL)
            self.l_SHARPE_RATIO_DAILY.append(SHARPE_RATIO_DAILY)
            self.l_SORTINO_RATIO_ANNUAL.append(SORTINO_RATIO_ANNUAL)
            self.l_SORTINO_RATIO_DAILY.append(SORTINO_RATIO_DAILY)
            self.l_MAX_DRAWDOWN.append(MAX_DRAWDOWN)
            self.l_PNL_MDD_RATIO.append(PNL_MDD_RATIO)
            self.l_WIN_RATE.append(WIN_RATE)
            self.l_WIN_RATE_SHORT.append(WIN_RATE_SHORT)
            self.l_WIN_RATE_LONG.append(WIN_RATE_LONG)

        self.MARGIN = self.START_MARGIN + self.TOTAL_PNL  # update margin after trade closes
        self.TOTAL_MARGIN = self.MARGIN



        # calculate performance metrics
        return

    def plots(self, rows=2, **kwargs):
    
        boxes=kwargs.get('boxes', False)
        trade_id=kwargs.get('trade_id', False)

        if rows < 2:
            stamp.critical("Strategy.plots(): rows must be >=2")
            return

        self.axs = fplt.create_plot(f"{self.MARKET_NAME}/{self.TIME_INTERVAL_STR} {self.df.index[0]} - {self.df.index[-1]}", rows=rows)

        # standard candles
        fplt.volume_ocv(self.df[['open', 'close', 'volume']], ax=self.axs[0].overlay())
        fplt.candlestick_ochl(self.df[['open', 'close', 'high', 'low']], ax=self.axs[0])
        
        # PnL chart
        if self.TOTAL_TRADES <= 0:
            log.warning("🚩 No trades were executed. Cannot plot PnL chart.")
            return
        else:
            fplt.add_line((self.df.index[0], self.START_MARGIN), (self.df.index[-1], self.START_MARGIN), ax=self.axs[-1], color="#130000", style="--")
            fplt.plot(self.df_cum_margin, ax=self.axs[-1], color="#ff6a00", legend="cumulative Pnl")
            plot_trades(tf=self.tf, cc=self.cc, timestep=self.TIME_INTERVAL, ax=self.axs[0], boxes=boxes, trade_id=trade_id)

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
            log.error(f"❌ Error calculating SL/TP prices: {e}")

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
            log.error(f"❌ Error calculating SL/TP prices: {e}")

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
                previous_margin = self.MARGIN

                self.LAST_RAW_PNL = tp_pnl
                self.LAST_PNL = tp_pnl-fee-spread
                self.LAST_CUMSUM_PNL += self.LAST_PNL
                self.MARGIN += self.LAST_PNL

                self.LAST_RETURN = self.LAST_PNL/previous_margin if previous_margin !=0 else 0
                self.l_DATETIME_RETURN.append(self.data['datetime'][i])

                if order_side =='buy':
                    self.WINS_LONG +=1
                    self.TOTAL_WINS +=1
                else: 
                    self.WINS_SHORT +=1
                    self.TOTAL_WINS +=1

                self.update_metrics() 

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
                                        'pnl': self.LAST_PNL,
                                        'margin': self.MARGIN,
                                        'return': self.LAST_RETURN,
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
                previous_margin = self.MARGIN

                self.LAST_RAW_PNL = sl_pnl
                self.LAST_PNL = sl_pnl-fee-spread
                self.LAST_CUMSUM_PNL += self.LAST_PNL
                self.MARGIN += self.LAST_PNL
                
                self.LAST_RETURN = self.LAST_PNL/previous_margin if previous_margin !=0 else 0
                self.l_DATETIME_RETURN.append(self.data['datetime'][i])

                self.update_metrics() 

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
                                        'fee': fee,
                                        'spread': spread,
                                        'pnl': self.LAST_PNL,
                                        'margin': self.MARGIN,
                                        'return': self.LAST_RETURN,
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

    
    # ====== Display functions ======

    def discord_stats(self):
        """Shows currents stats"""
        
        title = "⚖️ Strategy Metrics"

        if self.TEST_MODE == Setting.MODE_TEST:
            mode = "🟡 TEST"
            color = "#E5FF00"
            self.TIME_END = datetime.now(ZoneInfo("Europe/London"))
            self.TIME_ELAPSED = self.TIME_END - self.TIME_START
            self.TIME_WORK_DAYS = np.busday_count(self.TIME_START.date(),self.TIME_END.date())
            

        elif self.TEST_MODE == Setting.MODE_LIVE:
            mode = "🟢 LIVE"
            color = "#16c60c"
            self.TIME_END = datetime.now(ZoneInfo("Europe/London"))
            self.TIME_ELAPSED = self.TIME_END - self.TIME_START
            self.TIME_WORK_DAYS = np.busday_count(self.TIME_START.date(),self.TIME_END.date())

        elif self.TEST_MODE == Setting.MODE_BACKTEST:
            mode = "🟣 BACKTEST"
            color = "#886ce4"
        else:
            mode = "⚫ NONE"
            color = "#2e2e2e"

        start_time_str = self.TIME_START.strftime("%d/%m/%y %H:%M %Z") if self.TIME_START is not None else "Unknown"
        end_time_str = self.TIME_END.strftime("%d/%m/%y %H:%M %Z") if self.TIME_END is not None else "Unknown"
        time_elapsed_str = str(self.TIME_ELAPSED).split('.')[0] if self.TIME_ELAPSED is not None else "Unknown"
        time_elapsed_work_days_str = f"{self.TIME_WORK_DAYS} work days" if self.TIME_WORK_DAYS is not None else "Unknown"

        column1 = (
            f"Mode: `{mode}`\n"
            f"Strategy: `{self.STRATEGY_NAME}`\n"
            f"Interval: `{self.TIME_INTERVAL_STR}`\n"
            f"Start time `{start_time_str}`\n"
            f"End time `{end_time_str}`\n"
            f"Elapsed time `{time_elapsed_str}`\n"
            f"work days `{time_elapsed_work_days_str}`\n"
            f"Open Trades: `{self.OPEN_TRADES}`\n"
            f"Total Trades: `{self.TOTAL_TRADES}`\n"
            f"Total Longs: `{self.TOTAL_LONGS}`\n"
            f"Total Shorts: `{self.TOTAL_SHORTS}`\n"
            f"Average Profit: `{self.AVERAGE_PROFIT:.2f} {self.TICK_CURRENCY}`\n"
            f"Average Loss: `{self.AVERAGE_LOSS:.2f} {self.TICK_CURRENCY}`\n"
            f"Average Return: `{self.AVERAGE_RETURN*100:.2f}%`\n"
            

        )

        column2 = (
            
            f"Win Rate: `{self.TOTAL_WIN_RATE*100:.2f}%`\n"
            f"Win Rate Long: `{self.TOTAL_WIN_RATE_LONG*100:.2f}%`\n"
            f"Win Rate Short: `{self.TOTAL_WIN_RATE_SHORT*100:.2f}%`\n"
            f"Total PnL: `{self.TOTAL_PNL:.2f} {self.TICK_CURRENCY}`\n"
            f"Margin: `{self.TOTAL_MARGIN:.2f} {self.TICK_CURRENCY}`\n"
            f"Total Return: `{self.TOTAL_RETURN*100:.2f}%`\n"
            f"Profit Factor: `{self.TOTAL_PROFIT_FACTOR:.2f}`\n"
            f"Max Drawdown: `{self.TOTAL_MAX_DRAWDOWN:.2f} {self.TICK_CURRENCY}`\n"
            f"Payoff Ratio: `{self.TOTAL_PAYOFF_RATIO:.2f}`\n"
            f"Sharpe (Annual): `{self.TOTAL_SHARPE_RATIO_ANNUAL:.2f}`\n"
            f"Sharpe (Daily): `{self.TOTAL_SHARPE_RATIO_DAILY:.2f}`\n"
            f"Sortino (Annual): `{self.TOTAL_SORTINO_RATIO_ANNUAL:.2f}`\n"
            f"Sortino (Daily): `{self.TOTAL_SORTINO_RATIO_DAILY:.2f}`\n"
            f"PnL/MDD Ratio: `{self.TOTAL_PNL_MDD_RATIO:.2f}`\n"
        )

        # ---- ADD PNL CHART ----
        buf = basic_graph(
                                np.arange(len(self.l_CUMSUM_PNL)),
                                self.l_CUMSUM_PNL,
                                xlabel="Trades",
                                ylabel="Pnl",
                                color=color,
                                discord =True
                            )
        
        embed = [{'value': column1, 'type': "text", 'inline': True, "title": ""}, 
                 {'value': column2, 'type': "text", 'inline': True, "title": ""},
                 {'value': buf,     'type': "file", 'inline': True, "title": ""}]

        return [embed, color, title]
          
    def discord_market(self):

        title = "💵 Market Info"
        color = "#0066FF"

        top_row = (f"Name: `{self.MARKET_NAME}`\n")

        column1 = (

                f"Symbol: `{self.MARKET_SYMBOL} `\n"
                f"Type: `{self.MARKET_TYPE}`\n"
                f"Candle Buffer: `{self.CANDLE_BUFFER}`\n"
                f"Poll Interval: `{self.POLL_INTERVAL}`\n"
                f"Time Frame: `{self.TIME_INTERVAL_STR}`\n"
                f"Tick Currency: `{self.TICK_CURRENCY}`\n"
                f"Tick Size: `{self.TICK_SIZE:.3f}`\n"
                f"Tick Price: `{self.TICK_PRICE:.3f} {self.TICK_CURRENCY}`\n"
                f"Tick Spread: `{self.TICK_SPREAD:.2f}`\n"
                f"Tick Slippage: `{self.TICK_SLIPPAGE:.2f}`\n"

            )
    
        column2 = (
                    f"Point Slippage: `{self.POINT_SLIPPAGE:.2f}`\n"
                    f"Point Leverage: `{self.POINT_LEVERAGE:.2f}`\n"
                    f"Lot Currency: `{self.LOT_CURRENCY}`\n"
                    f"Lot min size: `{self.LOT_MIN_SIZE:.2f}`\n"
                    f"Lot increment: `{self.LOT_INCREMENT:.2f}`\n"
                    f"Fee Type: `{self.FEE_TYPE}`\n"
                    f"Fee: `{self.FEE:.2f} {self.TICK_CURRENCY}`\n"
                    f"Starting Margin: `{self.START_MARGIN:.2f} {self.TICK_CURRENCY}`\n"
                    f"Active Margin: `{self.TOTAL_MARGIN:.2f} {self.TICK_CURRENCY}`\n"
            )

        embed = [{'value': top_row, 'type': "text", 'inline': False, "title": ""}, 
                 {'value': column1, 'type': "text", 'inline': True, "title": ""},
                 {'value': column2, 'type': "text", 'inline': True, "title": ""}]

        return [embed, color, title]

    def discord_settings(self):

        title = "⚙️ Strategy Settings"

        if self.TEST_MODE == Setting.MODE_TEST:
            mode = "🟡"
            color = "#E5FF00"
            
        elif self.TEST_MODE == Setting.MODE_LIVE:
            mode = "🟢"
            color = "#16c60c"

        elif self.TEST_MODE == Setting.MODE_BACKTEST:
            mode = "🟣"
            color = "#886ce4"
        else:
            mode = "⚫"
            color = "#2e2e2e"

        inputs = ""
        for key, value in self.INPUT_PARAMS.items():
            inputs += f"{key}: `{value}`\n"

        settings = (
                    f"Test Mode: `{mode}{self.TEST_MODE}`\n"
                    f"Slippage Entry Mode: `{self.setting_slippage_entry}`\n"
                    f"Slippage SL Mode: `{self.setting_slippage_sl}`\n"
                    f"Slippage TP Mode: `{self.setting_slippage_tp}`\n"
                    f"Price Round Mode: `{self.setting_rounding_method}`\n"
                    f"Track all metrics history: `{self.setting_track_all_metrics}`\n"
                    
                )
        
        derived_settings = (
                    f"Rounding Mode: buy: `{self._setting_rm_buy}`\n"
                    f"Rounding Mode: buy|sl: `{self._setting_rm_buy_sl}`\n"
                    f"Rounding Mode: buy|tp: `{self._setting_rm_buy_tp}`\n"
                    f"Rounding Mode: sell: `{self._setting_rm_sell}`\n"
                    f"Rounding Mode: sell|sl: `{self._setting_rm_sell_sl}`\n"
                    f"Rounding Mode: sell|tp: `{self._setting_rm_sell_tp}`\n"
                    f"Candle buffer: `{self.CANDLE_BUFFER} candles`\n"
                    f"console listen time: `{self.setting_listen_time}s`\n"
                    f"Data poll interval: `{self.POLL_INTERVAL}s`\n"
                )
        
        embed = [ {'value': inputs, 'type': "text", 'inline': False, "title": "Modified Variables"},
                  {'value': settings, 'type': "text", 'inline': False, "title": "Settings"},
                  {'value': derived_settings, 'type': "text", 'inline': False, "title": "Hidden Settings"}]

        return [embed, color, title]

    def discord_inputs(self):

        title = "🕹️ Initial Inputs"
        color = "#C50D0D"

        inputs = ""
        for key, value in self.INPUT_PARAMS.items():
            inputs += f"{key}: `{value}`\n"

        embed = [ {'value': inputs, 'type': "text", 'inline': False, "title": ""}]

        return [embed, color, title]
    
    # ====== INTERNAL FUNCTIONS ======
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

        if method == Setting.ROUND:
            return round(price / self.TICK_SIZE) * self.TICK_SIZE
        elif method == Setting.FLOOR:
            return (price // self.TICK_SIZE) * self.TICK_SIZE
        elif method == Setting.CEIL:
            return (-( -price // self.TICK_SIZE)) * self.TICK_SIZE
        else:
            raise ValueError(f"Unknown rounding method: {method}")

    @final
    def _slippage(self, setting):

        if setting == Setting.SLIP_OFF:
            return 0.0
        elif setting == Setting.SLIP_WORST_CASE:
            return self.POINT_SLIPPAGE
        elif setting == Setting.SLIP_RANDOM:
            return self._round_to_tick(np.random.uniform(-self.POINT_SLIPPAGE, self.POINT_SLIPPAGE), Setting.ROUND)

    @final
    def _check_qty(self, qty):

        if qty < self.LOT_MIN_SIZE:
            log.warning(f"🚧 Order qty {qty} is less than min lot size {self.LOT_MIN_SIZE}")  
            return 0
        if not math.isclose(qty % self.LOT_INCREMENT, 0, abs_tol=9e-01):
            log.warning(f"🚧 Order qty {qty} not compatible with lot increment {self.LOT_INCREMENT}")
            return 0
        return 1

    
    


if __name__ == "__main__":

    qty = 10.2
    increment = 0.1
    print(qty % increment)
    print(math.isclose(qty % increment, 0, abs_tol=9e-01))
    pass


    
    



