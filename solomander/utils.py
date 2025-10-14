import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import colorsys

try:
    from .logger import log, stamp, pront
except ImportError:
    from logger import log, stamp, pront
  


def sharpe(returns, dates, mode="annual"):

    """Compute Sharpe Ratio from lists of returns and datetimes."""

    returns = np.array(returns, dtype=float)
    dates = pd.to_datetime(dates)

    # group by day manually (fast and memory-light)
    unique_days, inv_idx = np.unique([d.date() for d in dates], return_inverse=True)
    daily_sums = np.zeros(len(unique_days))
    np.add.at(daily_sums, inv_idx, returns)  # sums returns by day in-place

    if len(daily_sums) < 2 or np.std(daily_sums, ddof=1) == 0:
        return 0

    sr_daily = np.mean(daily_sums) / np.std(daily_sums, ddof=1)

    return sr_daily if mode == "daily" else sr_daily * np.sqrt(256)

def sortino(returns, dates, mode="annual"):
    """
    Compute Sortino ratio from lists of returns and datetimes.
    """

    # ---- Safety checks ----
    
    returns = np.array(returns, dtype=float)
    dates = pd.to_datetime(dates)

    # ---- Group by date (vectorized, like before) ----
    unique_days, inv_idx = np.unique([d.date() for d in dates], return_inverse=True)
    daily_sums = np.zeros(len(unique_days))
    np.add.at(daily_sums, inv_idx, returns)  # sum up returns by day

    if len(daily_sums) < 2:
        return 0

    # ---- Calculate downside deviation ----
    downside = daily_sums[daily_sums < 0]

    if downside.size < 2:
        return 0

    # ---- Compute daily Sortino ratio ----
    sortino_daily = np.mean(daily_sums) / np.std(downside, ddof=1)

    # ---- Return daily or annualized ----
    return sortino_daily if mode == "daily" else sortino_daily * np.sqrt(256)


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

def adjust_opacity(color, factor=0.1):
    
    """
    Reduces (or increases) the opacity of an 8-digit hex color.
    
    Args:
        color (str): 8-digit hex color string (e.g. '#0c57e497' or '#FF00FF80').
        factor (float): multiplier for opacity (0.0–1.0).
                        e.g. 0.5 makes it 50% more transparent.
                        
    Returns:
        str: New 8-digit hex color string with adjusted alpha.
    """

    color = color.lstrip("#")
    if len(color) not in (6, 8):
        raise ValueError("Color must be a 6- or 8-digit hex string.")

    # Extract RGB and alpha
    rgb = color[:6]
    alpha = color[6:] if len(color) == 8 else "FF"  # default opaque if not given

    # Convert alpha to int, apply factor
    new_alpha = int(int(alpha, 16) * factor)
    new_alpha = max(0, min(255, new_alpha))  # clamp between 0–255

    return f"#{rgb}{new_alpha:02x}"

def shift_hue(color: str, shift: float) -> str:
    """
    Shift a hex color's hue by a given amount (0–360 degrees), wrapping around.
    Args:
        color (str): Hex color, e.g. "#ff00ff" or "#ff00ffaa".
        shift (float or int): Hue shift amount in degrees.
    Returns:
        str: New hex color string (no alpha).
    """
    color = color.lstrip("#")

    # Parse RGB
    if len(color) == 8:
        r, g, b, a = [int(color[i:i+2], 16)/255 for i in (0, 2, 4, 6)]
    else:
        r, g, b = [int(color[i:i+2], 16)/255 for i in (0, 2, 4)]
        a = None

    # Convert to HLS
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Apply hue shift (wrap around 1.0)
    h = (h + shift / 360.0) % 1.0

    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    if a is not None:
        return "#{:02x}{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255), int(a * 255)
        )
    else:
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

def timedelta_to_str(td: pd.Timedelta) -> str:

    minutes = int(td.total_seconds() // 60)
    if minutes % 60 == 0:   # exact hours
        return f"{minutes//60}h"
    elif minutes % 1440 == 0:  # exact days
        return f"{minutes//1440}d"
    else:
        return f"{minutes}m"
    
def print_boxed_title(title):
    
    line = "═" * (len(title) + 4)
    pront.info(f"╔{line}╗")
    pront.info(f"║  {title}  ║")
    pront.info(f"╚{line}╝")





if __name__ == "__main__":


    returns = np.random.uniform(-0.04, 0.045, 24*256)
    dates = pd.date_range(start='2023-01-01', periods=len(returns), freq='H')

    print("Sharpe Ratio (Annual):", sharpe(returns, dates, mode="annual"))
    pass