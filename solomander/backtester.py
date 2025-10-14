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
    from .baseStrategy import Strategy
except ImportError:
    from indicators import vwap, timeband, sessions
    from logger import log, stamp, pront
    from data import load_yfinance
    from utils import max_drawdown, sharpe, sortino, timedelta_to_str, print_boxed_title
    from visuals import plot_trades
    from solomander.baseStrategy import Strategy




class _BacktesterStrategy:

    def __init__(self, strategy_instance: Strategy, df: pd.DataFrame, symbol_info: dict):
        
        self.s = strategy_instance                      # keep the original instance
        self.s.df = df.copy()
        self.s.symbol_data = symbol_info.copy()
    
        self.s._update_data_arrays()  # initial update of data arrays

        # Backtest-only fields
        self.s.TIME_INTERVAL     = self.s.df.index.to_series().diff().dropna().min()
        self.s.TIME_INTERVAL_STR = timedelta_to_str(self.s.TIME_INTERVAL)
        self.s.TEST_DAYS         = max(1, np.busday_count(self.s.df.index[0].date(), self.s.df.index[-1].date()))

        # any market params you want (store them here; use in wrappers as needed)
        self.s.symbol_data        = symbol_info.copy()
        self.s.MARKET_SYMBOL      = symbol_info.get('symbol', 'Unknown Symbol')
        self.s.MARKET_NAME        = symbol_info.get('name', 'Unknown Market')
        self.s.MARKET_TYPE        = symbol_info.get('type', 'futures')           # spot or futures
        self.s.TICK_CURRENCY      = symbol_info.get('tick_currency', 'USD')      # tick currency
        self.s.TICK_SIZE          = symbol_info.get('tick_size', 0.25)           # minimum price increment
        self.s.TICK_PRICE         = symbol_info.get('tick_price', 0.25)          # minimum price increment
        self.s.POINT_SLIPPAGE     = symbol_info.get('point_slippage', 1.0)
        self.s.TICK_SPREAD        = symbol_info.get('tick_spread', 0)            # typical spread in ticks
        self.s.LOT_CURRENCY      = symbol_info.get('lot_currency', 'USD')        # lot currency
        self.s.LOT_MIN_SIZE      = symbol_info.get('lot_min_size', 1)            # min contract size
        self.s.LOT_INCREMENT     = symbol_info.get('lot_increment', 1)           # minimum order size increment
        self.s.FEE_TYPE          = symbol_info.get('fee_type', 'fixed')          # fee round trip per trade
        self.s.FEE               = symbol_info.get('fee_value', 1.74)  

        # recalculate derived params
        self.s.MARGIN         = self.s.START_MARGIN
        self.s.POINT_LEVERAGE = self.s.TICK_PRICE / self.s.TICK_SIZE
        self.s.TICK_SLIPPAGE  = self.s.POINT_SLIPPAGE / self.s.TICK_SIZE

        # rebinding functions

        self.buy_bracket_orginal = self.s.buy_bracket
        self.sell_bracket_orginal = self.s.sell_bracket

        self.s.buy_bracket = self.buy_bracket.__get__(self.s, self.s.__class__)
        self.s.sell_bracket = self.sell_bracket.__get__(self.s, self.s.__class__)

        #log.debug("BacktesterStrategy wrapper instance created, strategy updated and wrapper methods added")
        pass

    # ------ OVERRIDDEN STRATEGY METHODS ------:

    @final
    def sell_bracket(self, i, qty, sl_price=None, tp_price=None, sl_pips=None, tp_pips=None, comments=''):
        
        self.buy_bracket_orginal(i, qty, sl_price, tp_price, sl_pips, tp_pips, comments)
        #stamp.info(f"test, SELL backtest wrapper")
        return 
        

    @final
    def buy_bracket(self, i, qty, sl_price=None, tp_price=None, sl_pips=None, tp_pips=None, comments=''):

        self.sell_bracket_orginal(i, qty, sl_price, tp_price, sl_pips, tp_pips, comments)
        #stamp.info(f"test, BUY backtest wrapper")
        return





class Backtester():

    def __init__(self, strategy_instance: Strategy, df: pd.DataFrame, symbol_info: dict, **kwargs):
        
        # update instance with backtester methods
        self.wrapper = _BacktesterStrategy(strategy_instance, df, symbol_info)  
       

    # ------ UNIQUE BACKTESTER METHODS ------
    @final
    def execute(self):

        # Fixed Strategy loop:
        for i in range(len(self.wrapper.s.df)):

            self.wrapper.s.loop_update(i)  

            if self.wrapper.s.buy_condition(i):

                self.wrapper.s.buy_action(i)
                pass

            if self.wrapper.s.sell_condition(i):

                self.wrapper.s.sell_action(i)
                pass

            self.wrapper.s._check_market_sltp(i)

        # CHECK IF ANY ORDERS WERE EXECUTED
        if self.wrapper.s.TOTAL_ORDERS == 0:
            log.warning("🚧 No orders were executed. Check your strategy logic.")
            return self.wrapper.s
        # recreate pandas from vectors (in case vectors have changed)
        #self.s.df = pd.DataFrame(self.s.data, index=self.s.data['datetime'])

        self.wrapper.s.tf = pd.DataFrame(self.wrapper.s.l_trades)
        self.wrapper.s.oo = pd.DataFrame(self.wrapper.s.l_orders_open)
        self.wrapper.s.cc = pd.DataFrame(self.wrapper.s.l_orders_closed)
        self.wrapper.s.df_cum_margin = pd.Series(data=self.wrapper.s.l_DATETIME_MARGIN,index=pd.to_datetime(self.wrapper.s.l_DATETIMES),name='margin')
        self.wrapper.s.df_cum_pnl = pd.Series(data=self.wrapper.s.l_DATETIME_PNL,index=pd.to_datetime(self.wrapper.s.l_DATETIMES),name='pnl')

        return self.wrapper.s

    @final
    def print_metrics(self):
        s = self.wrapper.s
        label_width = 22  # widen slightly for neat alignment

        print_boxed_title("INPUT PARAMETERS")
        for key, value in s.INPUT_PARAMS.items():
            pront.info(f"{key + ':':<{label_width}} {value}")

        print_boxed_title("SETTINGS")
        pront.info(f"{'Slippage Entry:':<{label_width}} {s.setting_slippage_entry}")
        pront.info(f"{'Slippage SL:':<{label_width}} {s.setting_slippage_sl}")
        pront.info(f"{'Slippage TP:':<{label_width}} {s.setting_slippage_tp}")
        pront.info(f"{'Rounding Method:':<{label_width}} {s.setting_rounding_method}")

        print_boxed_title("MARKET PARAMETERS")
        pront.info(f"{'Market Symbol:':<{label_width}} {s.MARKET_SYMBOL}")
        pront.info(f"{'Market Name:':<{label_width}} {s.MARKET_NAME}")
        pront.info(f"{'Market Type:':<{label_width}} {s.MARKET_TYPE}")
        pront.info(f"{'Tick Currency:':<{label_width}} {s.TICK_CURRENCY}")
        pront.info(f"{'Tick Size:':<{label_width}} {s.TICK_SIZE}")
        pront.info(f"{'Tick Price:':<{label_width}} {s.TICK_PRICE:.5f}")
        pront.info(f"{'Tick Spread:':<{label_width}} {s.TICK_SPREAD}")
        pront.info(f"{'Spread per Lot:':<{label_width}} {(s.TICK_SPREAD * s.TICK_PRICE):.3f} {s.TICK_CURRENCY}")
        pront.info(f"{'Point Slippage:':<{label_width}} {s.POINT_SLIPPAGE}")
        pront.info(f"{'Slippage per Lot:':<{label_width}} {(s.POINT_SLIPPAGE * s.POINT_LEVERAGE):.3f} {s.TICK_CURRENCY}")
        pront.info(f"{'Point Leverage:':<{label_width}} {s.POINT_LEVERAGE:.3f}")
        pront.info(f"{'Lot Currency:':<{label_width}} {s.LOT_CURRENCY}")
        pront.info(f"{'Lot Min Size:':<{label_width}} {s.LOT_MIN_SIZE}")
        pront.info(f"{'Lot Increment:':<{label_width}} {s.LOT_INCREMENT}")
        pront.info(f"{'Fee Type:':<{label_width}} {s.FEE_TYPE}")
        pront.info(f"{'Fee Value:':<{label_width}} {s.FEE}")
        pront.info(f"{'Start Margin:':<{label_width}} {s.START_MARGIN:.2f} {s.TICK_CURRENCY}")
        pront.info(f"{'Active Margin:':<{label_width}} {s.TOTAL_MARGIN:.2f} {s.TICK_CURRENCY}")

        print_boxed_title("TEST METRICS")
        pront.info(f"{'Start Date:':<{label_width}} {s.df.index[0]}")
        pront.info(f"{'End Date:':<{label_width}} {s.df.index[-1]}")
        pront.info(f"{'Total Days:':<{label_width}} {s.TEST_DAYS}")
        pront.info(f"{'Total Trades:':<{label_width}} {s.TOTAL_TRADES}")
        pront.info(f"{'Total Longs:':<{label_width}} {s.TOTAL_LONGS}")
        pront.info(f"{'Total Shorts:':<{label_width}} {s.TOTAL_SHORTS}")
        pront.info(f"{'Win Rate Total:':<{label_width}} {s.TOTAL_WIN_RATE * 100:.2f}%")
        pront.info(f"{'Win Rate Long:':<{label_width}} {s.TOTAL_WIN_RATE_LONG * 100:.2f}%")
        pront.info(f"{'Win Rate Short:':<{label_width}} {s.TOTAL_WIN_RATE_SHORT * 100:.2f}%")

        print_boxed_title("PERFORMANCE METRICS")
        pront.info(f"{'Total PnL:':<{label_width}} ${s.TOTAL_PNL:.2f}")
        pront.info(f"{'Total Return:':<{label_width}} {s.TOTAL_RETURN * 100:.2f}%")
        pront.info(f"{'Average Profit:':<{label_width}} ${s.AVERAGE_PROFIT:.2f}")
        pront.info(f"{'Average Loss:':<{label_width}} ${s.AVERAGE_LOSS:.2f}")
        pront.info(f"{'Average Return:':<{label_width}} {s.AVERAGE_RETURN * 100:.2f}%")
        pront.info(f"{'Total Profit Factor:':<{label_width}} {s.TOTAL_PROFIT_FACTOR:.2f}")
        pront.info(f"{'Total Payoff Ratio:':<{label_width}} {s.TOTAL_PAYOFF_RATIO:.2f}")
        pront.info(f"{'Total Max Drawdown:':<{label_width}} ${s.TOTAL_MAX_DRAWDOWN:.2f}")
        pront.info(f"{'PnL/MDD Ratio:':<{label_width}} {s.TOTAL_PNL_MDD_RATIO:.2f}")

        print_boxed_title("RISK METRICS")
        pront.info(f"{'Sharpe Ratio (Daily):':<{label_width}} {s.TOTAL_SHARPE_RATIO_DAILY:.2f}")
        pront.info(f"{'Sharpe Ratio (Annual):':<{label_width}} {s.TOTAL_SHARPE_RATIO_ANNUAL:.2f}")
        pront.info(f"{'Sortino Ratio (Daily):':<{label_width}} {s.TOTAL_SORTINO_RATIO_DAILY:.2f}")
        pront.info(f"{'Sortino Ratio (Annual):':<{label_width}} {s.TOTAL_SORTINO_RATIO_ANNUAL:.2f}")

        pront.info("\n")

    @final
    def show(self, **kwargs):
        self.wrapper.s.plots(**kwargs)
        fplt.show()
        return

