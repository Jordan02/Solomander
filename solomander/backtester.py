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
    from .strategy import Strategy
except ImportError:
    from indicators import vwap, timeband, sessions
    from logger import log, stamp, pront
    from data import load_yfinance
    from utils import max_drawdown, sharpe, sortino, timedelta_to_str, print_boxed_title
    from visuals import plot_trades
    from strategy import Strategy


class Backtester(Strategy):

    def __init__(self, df: pd.DataFrame, symbol: dict, **kwargs):
        super().__init__(df, symbol)

        # ----- NEW PARAMETERS -----
        self.TIME_INTERVAL     = df.index.to_series().diff().dropna().min()
        self.TIME_INTERVAL_STR = timedelta_to_str(self.TIME_INTERVAL)
        self.TEST_DAYS         = max(1, np.busday_count(df.index[0].date(), df.index[-1].date())) # number of business days in test period

        # ----- All KWARGS STORED AS PARAMS -----
        self.INPUT_PARAMS = kwargs
        self.init_kwargs = kwargs.copy() # store original kwargs for reference
        for key, value in kwargs.items():
            setattr(self, key, value)

        # ----- UPDATES REQUIRED AFTER KWARGS GIVEN -----
        self.update_data()
        self.ACTIVE_MARGIN = self.START_MARGIN  # starting margin
        self._setting_rm_buy      = "ceil" if self.setting_rounding_method == "worst_case" else "round"
        self._setting_rm_buy_sl   = "floor" if self.setting_rounding_method == "worst_case" else "round"
        self._setting_rm_buy_tp   = "floor" if self.setting_rounding_method == "worst_case" else "round"

        self._setting_rm_sell     = "floor" if self.setting_rounding_method == "worst_case" else "round"
        self._setting_rm_sell_sl  = "ceil" if self.setting_rounding_method == "worst_case" else "round"
        self._setting_rm_sell_tp  = "ceil" if self.setting_rounding_method == "worst_case" else "round"


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
        plot_trades(tf=self.tf, cc=self.cc, timestep=self.TIME_INTERVAL, ax=self.axs[0], boxes=boxes, trade_id=trade_id)
        
        # PnL chart
        fplt.add_line((self.df.index[0], self.START_MARGIN), (self.df.index[-1], self.START_MARGIN), ax=self.axs[-1], color="#130000", style="--")
        fplt.plot(self.df_cum_margin, ax=self.axs[-1], color="#ff6a00", legend="cumulative Pnl")

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
            self.l_cum_margin.append(self.ACTIVE_MARGIN)  # track margin over time

        # CHECK IF ANY ORDERS WERE EXECUTED
        if self.TOTAL_ORDERS == 0:
            log.warning("No orders were executed. Check your strategy logic.")
            return
        # recreate pandas from vectors (in case vectors have changed)
        self.df = pd.DataFrame(self.data, index=self.data['datetime'])
        self.tf = pd.DataFrame(self.l_trades)
        self.oo = pd.DataFrame(self.l_orders_open)
        self.cc = pd.DataFrame(self.l_orders_closed)
        self.df_cum_margin = pd.DataFrame({'datetime': self.data['datetime'], 'margin': self.l_cum_margin}).set_index('datetime')
        self.df_cum_pnl = self.tf[['pnl']].cumsum().rename(columns={'pnl':'cum_pnl'}).shift(1).fillna(0)
        
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
    def show(self, **kwargs):
        # CHECK IF ANY ORDERS WERE EXECUTED
        if self.TOTAL_ORDERS == 0:
            log.warning("No orders were executed. Check your strategy logic.")
            return
        
        self.plots(**kwargs)
        fplt.show()
        return

    @final
    def print_metrics(self):

        label_width = 18  # adjust so colons line up
        
        print_boxed_title("STRATEGY METRICS")
        for key, value in self.INPUT_PARAMS.items():
            pront.info(f"{key + ':':<{label_width}} {value}")

        print_boxed_title("SETTINGS")
        pront.info(f"{'Slippage Entry:':<{label_width}} {self.setting_slippage_entry}")
        pront.info(f"{'Slippage SL:':<{label_width}} {self.setting_slippage_sl}")
        pront.info(f"{'Slippage TP:':<{label_width}} {self.setting_slippage_tp}")
        pront.info(f"{'Rounding Method:':<{label_width}} {self.setting_rounding_method}")

        print_boxed_title("MARKET PARAMS")
        pront.info(f"{'Market Symbol:':<{label_width}} {self.MARKET_SYMBOL}")
        pront.info(f"{'Market Name:':<{label_width}} {self.MARKET_NAME}")
        pront.info(f"{'Market Type:':<{label_width}} {self.MARKET_TYPE}")
        pront.info(f"{'Tick Currency:':<{label_width}} {self.TICK_CURRENCY}")
        pront.info(f"{'Tick Size:':<{label_width}} {self.TICK_SIZE}")
        pront.info(f"{'Tick Price:':<{label_width}} {self.TICK_PRICE:.5f}")
        pront.info(f"{'Tick Spread:':<{label_width}} {self.TICK_SPREAD}")
        pront.info(f"{'Spread per lot:':<{label_width}} {self.TICK_SPREAD*self.TICK_PRICE:.3f} {self.TICK_CURRENCY}")
        pront.info(f"{'Point Slippage:':<{label_width}} {self.POINT_SLIPPAGE}")
        pront.info(f"{'Slippage per lot:':<{label_width}} {self.POINT_SLIPPAGE*self.POINT_LEVERAGE:.3f} {self.TICK_CURRENCY}")
        pront.info(f"{'Point Leverage:':<{label_width}} {self.POINT_LEVERAGE:.3f}")
        pront.info(f"{'Lot Currency:':<{label_width}} {self.LOT_CURRENCY}")
        pront.info(f"{'Lot Min Size:':<{label_width}} {self.LOT_MIN_SIZE}")
        pront.info(f"{'Lot Increment:':<{label_width}} {self.LOT_INCREMENT}")
        pront.info(f"{'Fee Type:':<{label_width}} {self.FEE_TYPE}")
        pront.info(f"{'Fee Value:':<{label_width}} {self.FEE}")
        pront.info(f"{'Start Margin:':<{label_width}} {self.START_MARGIN:.2f} {self.TICK_CURRENCY}")
        pront.info(f"{'Active Margin:':<{label_width}} {self.ACTIVE_MARGIN:.2f} {self.TICK_CURRENCY}")

        print_boxed_title("TEST METRICS")
        pront.info(f"{'Start:':<{label_width}} {self.df.index[0]}")
        pront.info(f"{'End:':<{label_width}} {self.df.index[-1]}")
        pront.info(f"{'Days:':<{label_width}} {self.TEST_DAYS}")
        pront.info(f"{'Total Trades:':<{label_width}} {self.TOTAL_TRADES}")
        pront.info(f"{'Total Longs:':<{label_width}} {self.TOTAL_LONGS}")
        pront.info(f"{'Total Shorts:':<{label_width}} {self.TOTAL_SHORTS}")

        print_boxed_title("STRATEGY METRICS")
        pront.info(f"{'PNL:':<{label_width}} ${self.PNL:.2f}")
        pront.info(f"{'Win Rate Long:':<{label_width}} {self.WIN_RATE_LONG:.2f}%")
        pront.info(f"{'Win Rate Short:':<{label_width}} {self.WIN_RATE_SHORT:.2f}%")
        pront.info(f"{'Win Rate Total:':<{label_width}} {self.WIN_RATE:.2f}%")
        pront.info(f"{'Average Profit:':<{label_width}} ${self.AVG_PROFIT:.2f}")
        pront.info(f"{'Average Loss:':<{label_width}} ${self.AVG_LOSS:.2f}")
        pront.info(f"{'Max Drawdown:':<{label_width}} ${self.MAX_DRAWDOWN:.2f}")
        pront.info(f"{'PnL/MDD Ratio:':<{label_width}} {self.PNL_MDD_RATIO:.2f}")
        pront.info(f"{'Payoff Ratio:':<{label_width}} {self.PAYOFF_RATIO:.2f}")
        pront.info(f"{'Profit Factor:':<{label_width}} {self.PROFIT_FACTOR:.2f}")
        pront.info(f"{'Sharpe Ratio(Y):':<{label_width}} {self.SHARPE_RATIO_ANNUAL:.2f}")
        pront.info(f"{'Sharpe Ratio(D):':<{label_width}} {self.SHARPE_RATIO_DAILY:.2f}")
        pront.info(f"{'Sortino Ratio(Y):':<{label_width}} {self.SORTINO_RATIO_ANNUAL:.2f}")
        pront.info(f"{'Sortino Ratio(D):':<{label_width}} {self.SORTINO_RATIO_DAILY:.2f}")
        pront.info("\n")
        return

        

