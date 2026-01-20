import asyncio
import yfinance as yf
import pandas as pd
import os
import ccxt
import time
import json
import MetaTrader5 as mt5
from dotenv import load_dotenv
import getpass
import threading
import time
import finplot as fplt
import talib.abstract as ta
from typing import final
import discord
from discord.ext import commands
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
from typing import Optional, Union


try: 
    from .logger import log, stamp, pront
    from .utils import timedelta_to_str, print_boxed_title
    from .data import read_symbol, _load_data, _write_data, tz_from_utx_offset, write_symbol
    from .baseStrategy import Strategy, Setting
    from .discordBot import DiscordBot
except ImportError: 
    #for running as main script
    from logger import log, stamp, pront
    from utils import timedelta_to_str, print_boxed_title
    from data import _load_data, _write_data, tz_from_utx_offset, write_symbol, read_symbol
    from baseStrategy import Strategy, Setting
    from discordBot import DiscordBot


MT5_TIMEFRAMES = {
    mt5.TIMEFRAME_M1: "1m",
    mt5.TIMEFRAME_M2: "2m",
    mt5.TIMEFRAME_M3: "3m",
    mt5.TIMEFRAME_M4: "4m",
    mt5.TIMEFRAME_M5: "5m",
    mt5.TIMEFRAME_M6: "6m",
    mt5.TIMEFRAME_M10: "10m",
    mt5.TIMEFRAME_M12: "12m",
    mt5.TIMEFRAME_M15: "15m",
    mt5.TIMEFRAME_M20: "20m",
    mt5.TIMEFRAME_M30: "30m",
    mt5.TIMEFRAME_H1: "1h",
    mt5.TIMEFRAME_H2: "2h",
    mt5.TIMEFRAME_H3: "3h",
    mt5.TIMEFRAME_H4: "4h",
    mt5.TIMEFRAME_H6: "6h",
    mt5.TIMEFRAME_H8: "8h",
    mt5.TIMEFRAME_H12: "12h",
    mt5.TIMEFRAME_D1: "1d",
    mt5.TIMEFRAME_W1: "1w",
    mt5.TIMEFRAME_MN1: "1mo"
}


def mt5_login() -> bool:

    # load MT5 env variables
    load_dotenv()  
    MT5_LOGIN = int(os.getenv('MT5_LOGIN'))
    MT5_PASSWORD = os.getenv('MT5_PASSWORD')
    MT5_SERVER = os.getenv('MT5_SERVER')

    # check they are set
    if not MT5_LOGIN or not MT5_PASSWORD or not MT5_SERVER:
        log.error("❌ MT5 login details not found in .env file.")
        return False
    
    login = mt5.initialize(login = MT5_LOGIN, server = MT5_SERVER, password = MT5_PASSWORD)

    # check login worked
    if not login:
        log.error(f"❌ MT5 initialization failed, error code = {mt5.last_error()}")
        return False
    
    stamp.success(f"✅ MT5 initialized successfully")
    
    return

def mt5_ensure_login():
    if mt5.account_info() is None:
        return 0
    return 1

def mt5_load_symbol(symbol: str, read=True, refresh_data=False):
    
    
    # ----- WE WANT TO READ -----
    
    if read:
        symbol_info = read_symbol(symbol)
        if symbol_info:
            stamp.success(f"✅ {symbol} info retreived from data_symbols.json")
            return symbol_info
        else:
            stamp.warning(f"🚧 {symbol} cannot be read from data_symbols.json, fetching now...")
    

    # ----- ENSURE LOGIN -----
    if not mt5_ensure_login():
        log.error("❌ MT5 is not initialized. Please call login_mt5() first.")
        return None
    
    # ----- FETCH DATA -----
    try:
        info = mt5.symbol_info(symbol)
        terminal_info = mt5.terminal_info()
    except Exception as e:
        log.error(f"❌ Error getting symbol info for {symbol}: {e}")
        return None
    
    # ----- infer timezone  -----
    if refresh_data:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1) # get 1 candle
        server_time = pd.to_datetime(rates[0]['time'], unit='s')
        utc_now = datetime.now(timezone.utc)
        offset = -int(round((utc_now.replace(tzinfo=None) - server_time).total_seconds() / 3600))
        tz_name, tz_utc_offset  = tz_from_utx_offset(offset)
        if abs(offset)>3:
            stamp.critical(f"{tz_utc_offset} is greater than 3hr shift: mt5 servers are typically in Europe/Moscow / UTC+3. Is it the weekend? Markets will not update on Weekends - Data is likely not syncing to time correctly. Check that volume spikes allign to NY open to validate.")


    symbol_info =  {
                    "symbol": info.name,
                    "data_tz": tz_name,
                    "data_tz_utc": tz_utc_offset,
                    "server": terminal_info.name,
                    "name": info.description,
                    "type": "CFD" if info.trade_calc_mode == 4 else "unknown",
                    "tick_currency": info.currency_profit,
                    "tick_size": info.point,
                    "tick_price": info.trade_tick_value,
                    "tick_spread": info.spread,               # add live spread
                    "point_slippage": 0.5,                # default to 0.5, can override depending on your broker
                    "lot_currency": "contracts",
                    "lot_increment": info.volume_step,
                    "lot_min_size": info.volume_min,
                    "fee_type": "fixed",                 # or override depending on your broker
                    "fee_value": 0,                      # FTMO CFDs usually spread only
                } 
    
    # IF WE WANT TO WRITE
    if refresh_data:
        write_success = write_symbol(symbol_info)
        if write_success:
            stamp.success(f"✅ {symbol} info successfully wrote to data_symbols.json")
        else:
            stamp.warning(f"🚧 {symbol} cannot be wrote to from data_symbols.json")

    return symbol_info



def mt5_hdata(symbol: str, mt5_time_interval, candle_lookback: int = 1000, candle_offset: int = 0, read=True, refresh_data=False) -> pd.DataFrame:

    
    # ----- CHECK IF FILE EXISTS -----

    interval_str = MT5_TIMEFRAMES.get(mt5_time_interval, "unknown")
    filename = f"mt5_{symbol}_{candle_lookback}c_{candle_offset}o_{interval_str}.csv".replace("-", "").replace("/", "-")
    
    if read == True:
        df = _load_data(filename) 
        if df is not None:
            stamp.success(f"✅ Data loaded from data/{filename} Opening now queen.")
            return df

    # ----- IF NOT RETREIVE DATA -----
    try:
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, mt5_time_interval, candle_offset, candle_lookback)
    except Exception as e:
        log.error(f"❌ MT5 download error {symbol}: {e}")
        return pd.DataFrame()

    # ----- FORMAT DATA -----
    symbol_data_tz_local = mt5_load_symbol(symbol,True,False)["data_tz"]
    df = _mt5_format_data(rates, symbol_data_tz_local)

    # ----- WRITE DATA -----
    if refresh_data == True:
        _write_data(df,filename)
        stamp.success(f"✅ {symbol} saved to data/{filename}. tz: {df.index.tz} {df.index[0].strftime("%d/%m/%y")} to {df.index[-1].strftime("%d/%m/%y")}, {mt5_time_interval} intervals, {candle_lookback} candles ")

    return df 

def mt5_server_info(name = 'name'):
    
    terminal_info = mt5.terminal_info()
    return terminal_info[name]

def _mt5_format_data(mt5_rates, tz_local) -> pd.DataFrame:
    
    # mt5_rates come in as numpy.ndarray with no timezone
    # THIS IS FINE, BACKTESTERS _UPDATE_DATA_ARRAYS will format data
    df = pd.DataFrame(mt5_rates)

    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={"time": "datetime", "tick_volume": "volume"}, inplace=True)
    df.set_index('datetime', inplace=True)
    df.index = df.index.tz_localize(tz_local)

    return df



class _mt5Strategy():
    
    def __init__(self, stratgy_instance: Strategy, symbol_info: dict, **kwargs):
        
        """
        TODO 
        - get actual account margin
        """
        self.s = stratgy_instance

        # New settings and parameters
        self.s.setting_slippage_entry = Setting.SLIP_OFF
        self.s.setting_slippage_sl = Setting.SLIP_OFF
        self.s.setting_slippage_tp = Setting.SLIP_OFF
        self.s.setting_rounding_method = Setting.ROUND_NEAREST
        
        # any market params you want (store them here; use in wrappers as needed)
        self.s.symbol_data        = symbol_info.copy()
        
        self.s.DATA_TZ            = symbol_info.get('data_tz', 'Unknown')
        self.s.DATA_TZ_UTC        = symbol_info.get('data_tz_utc', 'Unknown')
        self.s.DATA_SERVER        = symbol_info.get('server', 'Unknown')
        
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

        self.s._setting_rm_buy      = Setting.CEIL if self.s.setting_rounding_method == Setting.ROUND_WORST_CASE else Setting.ROUND
        self.s._setting_rm_buy_sl   = Setting.FLOOR if self.s.setting_rounding_method == Setting.ROUND_WORST_CASE else Setting.ROUND
        self.s._setting_rm_buy_tp   = Setting.FLOOR if self.s.setting_rounding_method == Setting.ROUND_WORST_CASE else Setting.ROUND
        self.s._setting_rm_sell     = Setting.FLOOR if self.s.setting_rounding_method == Setting.ROUND_WORST_CASE else Setting.ROUND
        self.s._setting_rm_sell_sl  = Setting.CEIL if self.s.setting_rounding_method == Setting.ROUND_WORST_CASE else Setting.ROUND
        self.s._setting_rm_sell_tp  = Setting.CEIL if self.s.setting_rounding_method == Setting.ROUND_WORST_CASE else Setting.ROUND

        # rebinding functions
        self.s.on_buy_bracket = self.on_buy_bracket.__get__(self.s,self.s.__class__)   
        self.s.on_sell_bracket = self.on_sell_bracket.__get__(self.s,self.s.__class__)   
        self.s.on_bracket_close_tp = self.on_bracket_close_tp.__get__(self.s,self.s.__class__)   
        self.s.on_bracket_close_sl = self.on_bracket_close_sl.__get__(self.s,self.s.__class__)   
        self.s.tp_sl_conditions = self.tp_sl_conditions.__get__(self.s,self.s.__class__)   

    
    @final
    def on_sell_bracket(self, trade_orders:list = None):

        # ---- recieving order details see basestrategy lists for order column details ----
        if trade_orders is None:
            log.error("No trade order information received")
            self.s.post_discord_message("No trade order information received")
            return trade_orders

        market, sl, tp = trade_orders
        _comments = market['comments']
        _id = market['trade_id']
        _qty = market['qty']
        _target_price = float(market['price'])
        _sl_price = sl['price']
        _tp_price = tp['price']
        _tp_pips = float(_target_price - _tp_price)
        _sl_pips = float(_sl_price - _target_price)

        # ---- using active tick data for entries -----
        tick = mt5.symbol_info_tick(self.s.MARKET_SYMBOL)
        if tick is None:
            log.error(f"🔽❌Tick unavailable (symbol_info_tick returned None)")
            self.s.post_discord_message(f"🔽❌Tick unavailable (symbol_info_tick returned None)")
            return trade_orders

        d = mt5.symbol_info(self.s.MARKET_SYMBOL).digits
        if d is None:
            log.warning(f"🔽❌no symbol info digit data, defaulting to 2.d.p")
            self.s.post_discord_message(f"🔽❌no symbol info digit data, defaulting to 2.d.p")
            d = 2

        # NOTE:
        # - For SELL you typically request at bid (tick.bid). Keeping this aligned helps drift/slip metrics make sense.
        # - Spread is still ask-bid.
        _exec_price = float(tick.bid)
        _spread = float(tick.ask - tick.bid)
        _drift = float(_target_price - _exec_price)

        # For SELL:
        # - SL should be ABOVE entry by _sl_pips
        # - TP should be BELOW entry by _tp_pips
        _sl = float(_exec_price + _sl_pips)
        _tp = float(_exec_price - _tp_pips)

        # ---- update order tickets -----
        market['price'] = _exec_price
        sl['price'] = _sl
        tp['price'] = _tp

        # ---- logging messgae details ----
        new_order_msg = (
            f"`🔔Order`  `🔽Sell`  "
            f"`{self.s.MODE_SYMBOL[self.s.setting_strategy_mode] + self.s.MODE_TEXT[self.s.setting_strategy_mode]}`  "
            f"`🆔{_id}`  `qty: {_qty}`  `target: {_target_price:.{d}f}`  `exec:{_exec_price:.{d}f}`   "
            f"`sl={_sl:.{d}f}`  `tp={_tp:.{d}f}`  "
            f"`spread:{_spread:.{d}f}`  `drift:{_drift:.{d}f}`"
        )
        embed = [{'value': new_order_msg, 'type': "text", 'inline': False, "title": ""}]
        color = self.s.MODE_COLOR[self.s.setting_strategy_mode]

        # ---- logging data for TEST_MODE ----
        if self.s.setting_strategy_mode is Setting.MODE_TEST:
            stamp.success(new_order_msg)
            self.s.post_discord_embed([embed, color, ""])
            return [market, sl, tp]

        # ---- Scheduling MT5 order for LIVE MODE ----
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.s.MARKET_SYMBOL,
            "volume": float(_qty),
            "type": mt5.ORDER_TYPE_SELL,
            "price": _exec_price,
            "sl": _sl,
            "tp": _tp,
            "deviation": self.s.DEVIATION,
            "magic": 123456,
            "comment": _comments,
            "type_filling": mt5.ORDER_FILLING_FOK,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"🔽❌ FAIL:" + new_order_msg)
            log.error(f"🔽❌ MT5 retcode: {result.retcode}")
            self.s.post_discord_message(f"🔽❌ FAIL:" + new_order_msg)
            self.s.post_discord_message(f"🔽❌ MT5 retcode: {result.retcode}")
        else:
            # For SELL, slip (worse fill) should be positive if filled LOWER than requested is bad?
            # Standardizing to "worse = positive":
            # requested = _exec_price (bid), fill = result.price
            # if fill is LOWER than requested on sell, that's worse (you sold cheaper) => positive slip = requested - fill
            _slip = _exec_price - result.price

            new_order_msg = new_order_msg + f"  `slip:{_slip:.{d}f}`"
            embed = [{'value': new_order_msg, 'type': "text", 'inline': False, "title": ""}]
            stamp.success(new_order_msg)
            self.s.post_discord_embed([embed, color, ""])

        return [market, sl, tp]

    @final
    def on_buy_bracket(self, trade_orders:list = None):

        # ---- recieving order details see basestrategy lists for order column detaials ----

        if trade_orders is None:
            log.error("No trade order information received")
            self.s.post_discord_message("No trade order information received")
            return trade_orders
        
        market, sl, tp = trade_orders
        _id = market['trade_id']
        _comments = market['comments']
        _qty = market['qty']
        _sl_price = sl['price']
        _tp_price = tp['price']
        _target_price = float(market['price'])
        _tp_pips = float(_tp_price - _target_price)
        _sl_pips = float(_target_price - _sl_price)

        # ---- using active tick data for entries -----

        tick = mt5.symbol_info_tick(self.s.MARKET_SYMBOL)
        if tick is None:
            log.error(f"🔼❌Tick unavailable (symbol_info_tick returned None)")
            self.s.post_discord_message(f"🔼❌Tick unavailable (symbol_info_tick returned None)")
            return trade_orders
        
        d = mt5.symbol_info(self.s.MARKET_SYMBOL).digits
        if d is None:
            log.warning(f"🔼❌no symbol info digit data, defaulting to 2.d.p")
            self.s.post_discord_message(f"🔼❌no symbol info digit data, defaulting to 2.d.p")
            d = 2

        _exec_price = float(tick.ask)
        _spread = float(tick.ask-tick.bid)
        _drift = float(_target_price-_exec_price)
        _sl = float(_exec_price - _sl_pips)
        _tp = float(_exec_price + _tp_pips)

        # ---- update order tickets -----

        market['price'] = _exec_price
        sl['price'] = _sl
        tp['price'] = _tp

        # ---- logging messgae details ----

        new_order_msg = f"`🔔Order`  `🔼buy`  `{self.s.MODE_SYMBOL[self.s.setting_strategy_mode] + self.s.MODE_TEXT[self.s.setting_strategy_mode]}`  `🆔{_id}`  `qty: {_qty}`  `target: {_target_price:.{d}f}`  `exec:{_exec_price:.{d}f}`   `sl={_sl:.{d}f}`  `tp={_tp:.{d}f}`   `spread:{_spread:.{d}f}`  `drift:{_drift:.{d}f}`"
        embed = [{'value': new_order_msg, 'type': "text", 'inline': False, "title": ""}]
        color = self.s.MODE_COLOR[self.s.setting_strategy_mode]

        # ---- logging data for TEST_MODE ----
        if self.s.setting_strategy_mode is Setting.MODE_TEST:
            stamp.success(new_order_msg)
            self.s.post_discord_embed([embed, color, ""])
            return [market, sl, tp]
    
        # ---- Scheduling MT5 order for LIVE MODE ----
        request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": self.s.MARKET_SYMBOL,
                        "volume": float(_qty),
                        "type": mt5.ORDER_TYPE_BUY,
                        "price": _exec_price,
                        "sl": _sl,
                        "tp": _tp,
                        "deviation": self.s.DEVIATION,
                        "magic": 123456,
                        "comment": _comments,
                        "type_filling": mt5.ORDER_FILLING_FOK,
                        "type_time": mt5.ORDER_TIME_GTC,
            }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"🔼❌ FAIL:" + new_order_msg)
            log.error(f"🔼❌ MT5 retcode: {result.retcode}")
            self.s.post_discord_message(f"🔼❌ FAIL:" + new_order_msg)
            self.s.post_discord_message(f"🔼❌ MT5 retcode: {result.retcode}")
        else:
            _slip = result.price - _exec_price
            new_order_msg = new_order_msg + f"  `slip:{_slip:.{d}f}`"
            embed = [{'value': new_order_msg, 'type': "text", 'inline': False, "title": ""}]
            stamp.success(new_order_msg)
            self.s.post_discord_embed([embed, color, ""])
        
        return  [market, sl, tp]

    @final
    def on_bracket_close_tp(self, trade_orders:list = None):


        """
        Callback used by live runners when a TP is hit.

        Base/backtest behaviour:
            - simply returns the orders unchanged (so the engine uses candle-derived prices)

        Live behaviour (MT5 runner):
            - fetch the latest tick
            - choose a realistic executable close price for the TP leg
            - update order_tp['price'] (and optionally its time)
            - keep everything else consistent so internal PnL uses the corrected exit
        """

        # ---- recieving order details see basestrategy lists for order column detaials ----
        if trade_orders is None:
            log.error("No trade order information received")
            self.s.post_discord_message("No trade order information received")
            return trade_orders

        order, order_sl, order_tp = trade_orders

        _id        = order.get('trade_id', None)
        _qty       = order.get('qty', None)
        _side      = order.get('side', None)          # open side: "buy" or "sell"
        _entry     = float(order.get('price', 0.0))
        _tp_target = float(order_tp.get('price', 0.0))

        # ---- using active tick data for close ----
        tick = mt5.symbol_info_tick(self.s.MARKET_SYMBOL)
        if tick is None:
            log.error("❌Tick unavailable (symbol_info_tick returned None)")
            self.s.post_discord_message("❌Tick unavailable (symbol_info_tick returned None)")
            return trade_orders

        info = mt5.symbol_info(self.s.MARKET_SYMBOL)

        # NOTE:
        # - digits controls display precision for logs (and can be used for rounding if desired).
        # - if symbol_info is unavailable, fall back to 2dp so discord logs don’t explode.
        d = getattr(info, "digits", None)
        if d is None:
            log.warning("❌no symbol info digit data, defauling to 2.d.p")
            self.s.post_discord_message("❌no symbol info digit data, defauling to 2.d.p")
            d = 2

        _bid = float(tick.bid)
        _ask = float(tick.ask)
        _spread = float(_ask - _bid)
        

        # NOTE:
        # When TP is hit, the *close* is the opposite side of the original position:
        # - If the position is a BUY (long), you close by SELLing -> executable at BID.
        # - If the position is a SELL (short), you close by BUYing -> executable at ASK.
        if _side == "buy":
            _exec_close = _bid
            _close_symbol = "🔽"
            _pnl = float(_exec_close - _entry)       # negative = loss
            _drift = float(_exec_close - _tp_target) # negative = loss
        else:
            _exec_close = _ask
            _close_symbol = "🔼"
            _pnl = float(_exec_close - _tp_target)    # negative = loss
            _drift = float(_tp_target-_exec_close)    # negative = loss


        # ---- update the TP order ticket used by internal accounting ----
        # We want your later logic to use order_tp['price'] as the exit price, so overwrite it here.
        order_tp['price'] = _exec_close

        # OPTIONAL:
        # If you want the closed order stamps to reflect "now", you can set these here,
        # but only do it if your calling code expects it (your loop already sets entry_time later).
        #
        # order_tp['entry_time'] = datetime.now()  # not recommended unless you standardize timezones

        # ---- logging (optional but consistent with your other callbacks) ----
        new_msg = (
            f"`🎯Tp hit`  `{_close_symbol}Close`   `{self.s.MODE_SYMBOL[self.s.setting_strategy_mode] + self.s.MODE_TEXT[self.s.setting_strategy_mode]}`  `🟩PNL:{self.s.TICK_CURRENCY}{_pnl:.{2}f}` "
            f"`🆔{_id}` `qty:{_qty}` "
            f"`entry:{_entry:.{d}f}` `tp_target:{_tp_target:.{d}f}` `tp_exec:{_exec_close:.{d}f}` "
            f"`spread:{_spread:.{d}f}` `drift:{_drift:.{d}f}`   `open: {self.s.OPEN_TRADES - 1}`  `total: {self.s.TOTAL_TRADES}`"
        )
        embed = [{'value': new_msg, 'type': "text", 'inline': False, "title": ""}]
        color = "#00FF00"

        # For backtests you might not want extra spam, but this mirrors your entry logging pattern.
        if self.s.setting_strategy_mode is Setting.MODE_TEST:
            stamp.success(new_msg)
            self.s.post_discord_embed([embed, color, ""])
            return [order, order_sl, order_tp]

        stamp.success(new_msg)
        self.s.post_discord_embed([embed, color, ""])

        return [order, order_sl, order_tp]
    
    @final
    def on_bracket_close_sl(self, trade_orders:list = None):

        """
        Callback used by live runners when an SL is hit.

        Base/backtest behaviour:
            - simply returns the orders unchanged (so the engine uses candle-derived prices)

        Live behaviour (MT5 runner):
            - fetch the latest tick
            - choose a realistic executable close price for the SL leg
            - update order_sl['price'] (and optionally its time)
            - keep everything else consistent so internal PnL uses the corrected exit
        """

        # ---- recieving order details see basestrategy lists for order column detaials ----
        if trade_orders is None:
            log.error("No trade order information received")
            self.s.post_discord_message("No trade order information received")
            return trade_orders

        order, order_sl, order_tp = trade_orders

        _id        = order.get('trade_id', None)
        _qty       = order.get('qty', None)
        _side      = order.get('side', None)          # open side: "buy" or "sell"
        _entry     = float(order.get('price', 0.0))
        _sl_target = float(order_sl.get('price', 0.0))

        # ---- using active tick data for close ----
        tick = mt5.symbol_info_tick(self.s.MARKET_SYMBOL)
        if tick is None:
            log.error("❌Tick unavailable (symbol_info_tick returned None)")
            self.s.post_discord_message("❌Tick unavailable (symbol_info_tick returned None)")
            return trade_orders

        info = mt5.symbol_info(self.s.MARKET_SYMBOL)

        # NOTE:
        # - digits controls display precision for logs (and can be used for rounding if desired).
        # - if symbol_info is unavailable, fall back to 2dp so discord logs don’t explode.
        d = getattr(info, "digits", None)
        if d is None:
            log.warning("❌no symbol info digit data, defauling to 2.d.p")
            self.s.post_discord_message("❌no symbol info digit data, defauling to 2.d.p")
            d = 2

        _bid = float(tick.bid)
        _ask = float(tick.ask)
        _spread = float(_ask - _bid)

        # NOTE:
        # When SL is hit, the *close* is the opposite side of the original position:
        # - If the position is a BUY (long), you close by SELLing -> executable at BID.
        # - If the position is a SELL (short), you close by BUYing -> executable at ASK.
        #
        # Since this is an SL hit, PnL should be negative.
        if _side == "buy":
            _exec_close = _bid
            _close_symbol = "🔽"
            _pnl = float(_exec_close - _entry)        # long loss => negative
            _drift = float( _exec_close - _sl_target) # negative = loss
        else:
            _exec_close = _ask
            _close_symbol = "🔼"
            _pnl = float(_entry - _exec_close)         # short loss => negative
            _drift = float( _sl_target - _exec_close)  # negative = loss

        # ---- update the SL order ticket used by internal accounting ----
        # We want your later logic to use order_sl['price'] as the exit price, so overwrite it here.
        order_sl['price'] = _exec_close

        # OPTIONAL:
        # If you want the closed order stamps to reflect "now", you can set these here,
        # but only do it if your calling code expects it (your loop already sets entry_time later).
        #
        # order_sl['entry_time'] = datetime.now()  # not recommended unless you standardize timezones

        # ---- logging (optional but consistent with your other callbacks) ----
        new_msg = (
            f"`🛑Sl hit`  `{_close_symbol}Close`   `{self.s.MODE_SYMBOL[self.s.setting_strategy_mode] + self.s.MODE_TEXT[self.s.setting_strategy_mode]}`  `🟥PNL:{self.s.TICK_CURRENCY}{_pnl:.{2}f}` "
            f"`🆔{_id}` `qty:{_qty}` "
            f"`entry:{_entry:.{d}f}` `sl_target:{_sl_target:.{d}f}` `sl_exec:{_exec_close:.{d}f}` "
            f"`spread:{_spread:.{d}f}` `drift:{_drift:.{d}f}`   `open: {self.s.OPEN_TRADES - 1}`  `total: {self.s.TOTAL_TRADES}`"
        )
        embed = [{'value': new_msg, 'type': "text", 'inline': False, "title": ""}]
        color = "#FF0000"

        # For backtests you might not want extra spam, but this mirrors your entry logging pattern.
        if self.s.setting_strategy_mode is Setting.MODE_TEST:
            stamp.success(new_msg)
            self.s.post_discord_embed([embed, color, ""])
            return [order, order_sl, order_tp]

        stamp.success(new_msg)
        self.s.post_discord_embed([embed, color, ""])

        return [order, order_sl, order_tp]

    @final
    def tp_sl_conditions(self, i, order_side, order_tp, order_sl):

        tick = mt5.symbol_info_tick(self.s.MARKET_SYMBOL)
        if tick is None:
            log.error("❌Tick unavailable (symbol_info_tick returned None, None)")
            self.s.post_discord_message("❌Tick unavailable (symbol_info_tick returned None, None)")
            return False, False

        bid = float(tick.bid)
        ask = float(tick.ask)

        if order_side == "buy":
            tp_condition = bid >= float(order_tp["price"])
            sl_condition = bid <= float(order_sl["price"])
        else:
            tp_condition = ask <= float(order_tp["price"])
            sl_condition = ask >= float(order_sl["price"])

        return tp_condition, sl_condition
        
class MT5_live(Strategy):

    def __init__(self, strategy_instance: Strategy, symbol_str = "US100.cash", timeframe = mt5.TIMEFRAME_M5, candle_buffer = 500, poll_interval = 5, test_mode = True, **kwargs):
        
        # ----- Runner params -----
        self.setting_password = str(os.getenv('MT5_BOT_PASSWORD', '123'))  # default password if not set in .env
        self._ask_password = threading.Event()
        self._ask_password.clear() #switch off
        self._password_verified = threading.Event()
        self._password_verified.clear() #switch off
        self.setting_listen_time = 0.25 #seconds

        self._running = threading.Event()
        self._running.set() #switch on
        self.discord_bot = None
        self._discord_attached = threading.Event()
        self._discord_attached.clear() #switch off
        self._bar_needs_closed = threading.Event()
        self._bar_needs_closed.clear()

        # Console listen thread for stop command
        self.___console_listener = threading.Thread(target=self._console_ear, daemon=True)
        self.___console_listener.start()

        # ----- Ask for password -----
        self._ask_password.set()
        stamp.input("Enter in password:")
        while self._password_verified.is_set() is False:
            time.sleep(self.setting_listen_time) 

        # ----- ENSURE MT5 LOGIN -----
        mt5_login()

        # ----- INITIALIZE WRAPPER -----
        self.w = _mt5Strategy(strategy_instance, symbol_info=mt5_load_symbol(symbol_str))
        self.s = self.w.s # strategy instance
                                    
        self.s.setting_strategy_mode = Setting.MODE_TEST if test_mode else Setting.MODE_LIVE
        self.s.TIME_INTERVAL = timeframe 
        self.s.TIME_INTERVAL_STR = MT5_TIMEFRAMES.get(self.s.TIME_INTERVAL, "unknown")
        self.s.CANDLE_BUFFER = candle_buffer
        self.s.POLL_INTERVAL = poll_interval
        self.s.TIME_START = datetime.now(ZoneInfo("Europe/London"))
        self.s.TIME_ELAPSED = None
        self.s.TIME_WORK_DAYS = None
        self.s.DEVIATION = 10
        self.s.setting_listen_time = self.setting_listen_time

        self.s.df = self._mt5_fetch_latest(2) # give intial data
         
        
    # ------ UNIQUE MT5 LIVE METHODS ------

    def _mt5_fetch_latest(self, candles: int=1) -> pd.DataFrame:

        # ----- RETRIEVE DATA -----
        try:
            rates = mt5.copy_rates_from_pos(self.s.MARKET_SYMBOL, self.s.TIME_INTERVAL, 0, candles)
        except Exception as e:
            log.error(f"❌ MT5 download error {self.s.MARKET_SYMBOL}: {e}")
            return pd.DataFrame()

        # ----- FORMAT DATA -----
        df= _mt5_format_data(rates, self.s.DATA_TZ)

        return df

    def _console_ear(self):

        # turn off console imput until password is set
        while self._password_verified.is_set() is False:
            time.sleep(self.setting_listen_time)  #prevent busy wait, be kind to cpu

            if self._ask_password.is_set():
                time.sleep(self.setting_listen_time)  #prevent busy wait, be kind to cpu
                cmd = input().strip().lower()
                if cmd == self.setting_password:
                    stamp.success("[CMD] ✅ Password correct")
                    self._password_verified.set()
                else:
                    stamp.error("[CMD] ❌ Password incorrect, try again.")


        while self._running.is_set():
            time.sleep(self.setting_listen_time)  #prevent busy wait, be kind to cpu
            cmd = input().strip().lower()
            if cmd == "stop":
                stamp.success("[CMD] ✅ Stop command received.")
                stamp.success("[CMD] 💤 Discord bot closing")
                self._running.clear()

    def _check_new_bar(self):

        """Check if a new bar has formed and reset flag when it does."""
        latest_candle_time = self.s.df.index[-1]  # current bar open time

        # intialise this variable
        if not hasattr(self, "_last_candle_time"):
            self._last_candle_time = latest_candle_time
            return

        # If the candle time has advanced, we know a new bar formed
        if latest_candle_time > self._last_candle_time:
            self._last_candle_time = latest_candle_time
            self._bar_needs_closed.clear()
            #stamp.success(f"[CANDLE] 🕒 New bar detected: {latest_candle_time}")

    def mt5_stream(self):

        # ----- udpate date -----
        self.s.df = self._mt5_fetch_latest(self.s.CANDLE_BUFFER)  # initial fetch to set up    
        self.s._update_data_arrays()
        
        # ----- is discord bot connected? -----
        if self._discord_attached.is_set() is False:
            stamp.warning("🚧 No Discord bot attached, proceeding without it.")
        else:   
            # pass to strategy
            self.s.DISCORD_BOT = self.discord_bot

        # ----- print input params -----
        label_width = 18 
        print_boxed_title("INPUT PARAMS")
        for key, value in self.s.INPUT_PARAMS.items():
            pront.info(f"{key + ':':<{label_width}} {value}")


        try:
            while self._running.is_set():

                next_time = time.time()+ self.s.POLL_INTERVAL
                
                # ----- LOOP LOGIC -----
                stamp.info(f"🔄 datetime: {self.s.df.index[-1].strftime('%H:%M:%S')}| close: {self.s.df['close'].iloc[-1]}| Open trades: {self.s.OPEN_TRADES} | Total trades: {self.s.TOTAL_TRADES}")
                
                self.s.loop_update(-1)
                self.s.df = self._mt5_fetch_latest(self.s.CANDLE_BUFFER)  # fetch latest data
                self.s._update_data_arrays()

                # buy logic (one trade per bar)
                if not self._bar_needs_closed.is_set():
                    if self.s.buy_condition(-1):
                        self.s.buy_action(-1)
                        self._bar_needs_closed.set()
                        pass

                if not self._bar_needs_closed.is_set():
                    if self.s.sell_condition(-1):
                        self.s.sell_action(-1)
                        self._bar_needs_closed.set()
                        pass
                
                self._check_new_bar() 
                self.s._check_market_sltp(-1)

                # ----- END LOOP LOGIC -----
                sleep_time = max(0, next_time - time.time())
                while  sleep_time > 0 and self._running.is_set():
                    time.sleep(min(self.setting_listen_time,sleep_time))
                    sleep_time = next_time- time.time()

        finally:
            #mt5.shutdown()
            stamp.success("🛑 Stopping MT5 live data stream...")
            self.s.post_discord_message("🛑 Stopping MT5 live data stream...")
    


        

if __name__ == "__main__":


    mt5_login()
    symbol = mt5_load_symbol("US100.cash", read=True, refresh_data=True)
    print(symbol)

   
