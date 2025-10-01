import yfinance as yf
import pandas as pd
import os
import ccxt
import time
import json
import MetaTrader5 as mt5
from dotenv import load_dotenv


try: 
    from .logger import log, stamp, pront
    from .utils import timedelta_to_str
    from .data import load_symbol, load_json, get_symbol_list, _load_data, _write_data
except ImportError: 
    #for running as main script
    from logger import log, stamp, pront
    from utils import timedelta_to_str
    from data import load_symbol, load_json, get_symbol_list, _load_data, _write_data


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
        log.error("❌ MT5 is not initialized. Please call login_mt5() first.")
        return 0
    return 1

def mt5_symbol_info(symbol: str):
    
    # exit if not logged in
    if not mt5_ensure_login():
        return None
    
    # exit if data already exisits
    symbol_info = load_symbol(symbol)
    if symbol_info is not None:
        stamp.success(f"✅ {symbol} info retreived from data_symbols.json")
        return symbol_info

    # otherwisetry to get it from MT5
    try:
        info = mt5.symbol_info(symbol)
    except Exception as e:
        log.error(f"❌ Error getting symbol info for {symbol}: {e}")
        return None

    symbol_info =  {
                    "symbol": info.name,
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
    
    json_path = os.path.join(os.path.dirname(__file__), "..", "data_symbols", "data_symbols.json")
    data = load_json(json_path)
    with open(json_path, "w") as f:
        data[symbol] = symbol_info
        json.dump(data, f, indent=4)

    stamp.success(f"✅ {symbol} info successfully downloaded MT5 to data_symbols.json")

    return symbol_info

def mt5_hdata(symbol: str, interval, lookback: int = 1000):

    # ----- ENSURE LOGIN -----
    if not mt5_ensure_login():
        return pd.DataFrame()
    
    # ----- CHECK IF THERE IS A VALID SYMBOL /META DATA FOR IT -----
    if load_symbol(symbol) is None:
        return pd.DataFrame()

    # ----- CHECK IF FILE EXISTS -----
    interval_str = MT5_TIMEFRAMES.get(interval, "unknown")
    filename = f"mt5_{symbol}_{lookback}c_{interval_str}.csv".replace("-", "").replace("/", "-")
    df = _load_data(filename) 
    if df is not None:
        stamp.success(f"✅ Data loaded from data/{filename} Opening now queen.")
        return df

    # ----- IF NOT DOWNLOAD DATA -----
    try:
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, interval, 0, lookback)
    except Exception as e:
        log.error(f"❌ MT5 download error {symbol}: {e}")
        return pd.DataFrame()

    # ----- FORMAT DATA -----
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={"time": "datetime", "tick_volume": "volume"}, inplace=True)
    df.set_index('datetime', inplace=True)
    df = df.tz_localize('UTC')

    # ----- WRITE DATA -----
    _write_data(df,filename)
    stamp.success(f"✅ {symbol} saved to data/{filename}. tz: {df.index.tz} {df.index[0].strftime("%d/%m/%y")} to {df.index[-1].strftime("%d/%m/%y")}, {interval} intervals, {lookback} candles ")

    return df 

if __name__ == "__main__":
    mt5_login()

    symbol = mt5_symbol_info("US100.cash")
    
    df = mt5_hdata("US100.cash", mt5.TIMEFRAME_M5, lookback=1000)

    print(MT5_TIMEFRAMES.get(mt5.TIMEFRAME_M5))
    print(df.index.tz)

