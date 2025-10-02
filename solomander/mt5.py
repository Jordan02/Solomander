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


try: 
    from .logger import log, stamp, pront
    from .utils import timedelta_to_str
    from .data import load_symbol, load_json, get_symbol_list, _load_data, _write_data
    from .strategy import Strategy
except ImportError: 
    #for running as main script
    from logger import log, stamp, pront
    from utils import timedelta_to_str
    from data import load_symbol, load_json, get_symbol_list, _load_data, _write_data
    from strategy import Strategy


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

def mt5_symbol_info(symbol: str):
    
    # ----- ENSURE LOGIN -----
    if not mt5_ensure_login():
        log.error("❌ MT5 is not initialized. Please call login_mt5() first.")
        return None
    
    # ----- READ IF ALREADY DOWLOADED -----
    symbol_info = load_symbol(symbol)
    if symbol_info is not None:
        stamp.success(f"✅ {symbol} info retreived from data_symbols.json")
        return symbol_info

    # ----- OTHERWISE DOWNLOAD AND SAVE -----
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

def mt5_hdata(symbol: str, mt5_time_interval, candle_lookback: int = 1000, candle_offset: int = 0) -> pd.DataFrame:

    # ----- ENSURE LOGIN -----
    if not mt5_ensure_login():
        log.error(f"❌ MT5 is not initialized. Please call login_mt5() first.")
        return pd.DataFrame()
    
    # ----- CHECK IF THERE IS A VALID SYMBOL /META DATA FOR IT -----
    if load_symbol(symbol) is None:
        return pd.DataFrame()

    # ----- CHECK IF FILE EXISTS -----
    interval_str = MT5_TIMEFRAMES.get(mt5_time_interval, "unknown")
    filename = f"mt5_{symbol}_{candle_lookback}c_{interval_str}.csv".replace("-", "").replace("/", "-")
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
    df = _mt5_format_data(rates)

    # ----- WRITE DATA -----
    _write_data(df,filename)
    stamp.success(f"✅ {symbol} saved to data/{filename}. tz: {df.index.tz} {df.index[0].strftime("%d/%m/%y")} to {df.index[-1].strftime("%d/%m/%y")}, {mt5_time_interval} intervals, {candle_lookback} candles ")

    return df 

def _mt5_format_data(mt5_rates) -> pd.DataFrame:
    
    df = pd.DataFrame(mt5_rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={"time": "datetime", "tick_volume": "volume"}, inplace=True)
    df.set_index('datetime', inplace=True)
    df = df.tz_localize('UTC')

    return df



class mt5_live(Strategy):

    def __init__(self, symbol_str, timeframe = mt5.TIMEFRAME_M5, candle_buffer = 500, poll_interval = 5, **kwargs):

        # ----- ENSURE LOGIN and symbol info -----
        mt5_login()

        # ----- VALIDATION CHECKS -----
        if load_symbol(symbol_str) is None:
            log.error(f"❌ {symbol_str} is not a valid symbol.")
        
        mt5.symbol_select(symbol_str, True)
        super().__init__(pd.DataFrame(), mt5_symbol_info(symbol_str))
        
        # ----- NEW PARAMETERS -----
        self.TIME_INTERVAL_MT5 = timeframe
        self.CANDLE_BUFFER = candle_buffer
        self.POLL_INTERVAL = poll_interval
        self.setting_slippage_entry = "off"
        self.setting_slippage_sl = "off"
        self.setting_slippage_tp = "off"
        self.setting_rounding_method = "nearest"
        self.setting_password = str(os.getenv('MT5_BOT_PASSWORD', '123'))  # default password if not set in .env
        
        # ---- THREADING -----
        self._running = threading.Event()         
        self.setting_listen_time = 0.25      # seconds between checking for stop command

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

    def _mt5_fetch_latest(self):

        # ----- RETRIEVE DATA -----
        try:
            rates = mt5.copy_rates_from_pos(self.MARKET_SYMBOL, self.TIME_INTERVAL_MT5, 0, self.CANDLE_BUFFER)
        except Exception as e:
            log.error(f"❌ MT5 download error {self.MARKET_SYMBOL}: {e}")
            return pd.DataFrame()

        # ----- FORMAT DATA -----
        df= _mt5_format_data(rates)

        return df

    def _input_running_listener(self):

        while self._running.is_set():
            time.sleep(self.setting_listen_time)  #prevent busy wait, be kind to cpu
            cmd = input().strip().lower()
            if cmd == "stop":
                stamp.success("✅ Stop command received.")
                self._running.clear()

        

    def mt5_stream(self):

        if not self.check_password():
            stamp.error("❌ Incorrect password. Access denied.")
            return 
        
        # ----- set up stop listener on seperate thread -----
        self._running.set() #switch on
        listener = threading.Thread(target=self._input_running_listener, daemon=True)
        listener.start()
        stamp.success("✅ Password correct. Generational wealth loading...") 
        
        
        try:
            while self._running.is_set():

                next_time = time.time()+ self.POLL_INTERVAL
                
                # ----- LOOP LOGIC -----
                self.df = self._mt5_fetch_latest()  # initial fetch to set up
                stamp.info(f"🔄 close: {self.df['close'].iloc[-1]}")

                
            
                # ----- END LOOP LOGIC -----
                
                sleep_time = max(0, next_time - time.time())
                while  sleep_time > 0 and self._running.is_set():
                    time.sleep(min(self.setting_listen_time,sleep_time))
                    sleep_time = next_time- time.time()

                   
        finally:
            #mt5.shutdown()
            stamp.success("🛑 Stopping MT5 live data stream...")
    
    
    def check_password(self):

        stamp.input("🔐 Please enter your password to start bot: ")
        password = getpass.getpass("")
        if password == self.setting_password: 
            return True
        else:
            return False

if __name__ == "__main__":

    bot1 = mt5_live("US100.cash", timeframe=mt5.TIMEFRAME_M5, candle_buffer=500, poll_interval=0.5)

    bot1.mt5_stream()

    print(bot1.POLL_INTERVAL)
    print(bot1.SHARPE_RATIO_ANNUAL)
