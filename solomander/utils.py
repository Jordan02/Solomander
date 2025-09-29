import pandas as pd
import numpy as np
import random

try:
    from .logger import log, stamp, pront
except ImportError:
    from logger import log, stamp, pront
  


def sharpe(trades:pd.DataFrame, type:str="annual"):
    """Sharpe Ratio

    trades must contain 'exit_time' (formatted as a datetime) and 'return' columns
    #note this is the shortcut verison. Fine, but as chat about it if you want.

    """
    if trades.empty or trades['return'].std(ddof=1) == 0:
        log.debug("SR: No trades or zero stddev on returns")
        return 0

    daily_returns = trades.groupby(trades['exit_time'].dt.date)['return'].sum()

    SR_Daily = (daily_returns.mean() / daily_returns.std(ddof=1))

    if type == "daily":
        return SR_Daily
    
    return SR_Daily * np.sqrt(252)  # assuming 252 trading days in a year

def sortino(trades: pd.DataFrame, type: str = "annual"):
    """Sortino Ratio

    trades must contain 'exit_time' and 'return' columns
    """

    if trades.empty or trades['return'].std(ddof=1) == 0:
        return 0

    daily_returns = trades.groupby(trades['exit_time'].dt.date)['return'].sum()
    downside = daily_returns[daily_returns < 0]

    sortino_daily = daily_returns.mean() / downside.std(ddof=1)

    if type == "daily":
        return sortino_daily

    return sortino_daily * np.sqrt(252)  # annualized

def max_drawdown(pnl):
    # ensure numpy array, but only convert if necessary
    if isinstance(pnl, pd.Series):
        pnl_array = pnl.to_numpy()
    elif isinstance(pnl, np.ndarray):
        pnl_array = pnl
    else:
        pnl_array = np.asarray(pnl)  # fallback for lists, etc.

    max_loss = 0
    current_loss = 0

    for val in pnl_array:
        if val < 0:
            current_loss += val  # accumulate drawdown
            max_loss = min(max_loss, current_loss)
        else:
            current_loss = 0  # reset on win

    return max_loss

def random_color(alpha=1.0):
    rgb = random.randint(0, 0xFFFFFF)
    a = int(alpha * 255)
    return "#{:06x}{:02x}".format(rgb, a)
