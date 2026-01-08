import finplot as fplt
import pandas as pd
import pyqtgraph as pg

import matplotlib
matplotlib.use("Agg")  # non-GUI backend (for servers / threads)
import matplotlib.pyplot as plt
import mplcyberpunk as cyberpunk
import io
import numpy as np


try:

    from .logger import log, stamp, pront
    from .data import yfin_load_data
    from .utils import load_graph_color, format_graph
except ImportError:
    
    from logger import log, stamp, pront
    from data import yfin_load_data
    from utils import load_graph_color, format_graph
    

"""
.	'.'	small dot (pixel-like)
o	'o'	circle (the one you’re using)
s	's'	square
t	't'	triangle (pointing up)
d	'd'	diamond
+	'+'	plus sign
x	'x'	cross
p	'p'	pentagon
h	'h'	hexagon

"""



def plot_trades(tf: pd.DataFrame, cc: pd.DataFrame, timestep, ax, boxes=False, trade_id=False):

    """ Plot trade on finplot ax from trade dataframe tf """
    
    if tf.empty:
        log.debug("trade tf is empty, nothing to plot")
        return
    if ax is None:
        log.debug("ax is None, cannot plot trade")
        return

    # buy and sell markers
    sell_markers = cc.loc[(cc['side']=='sell') & (cc['filled'] != 0), ['entry_time','price']].reset_index(drop=True)
    buy_markers  = cc.loc[(cc['side']=='buy')  & (cc['filled'] != 0), ['entry_time','price']].reset_index(drop=True)
    
    for i in sell_markers.index:
        fplt.add_text((sell_markers.loc[i,'entry_time'], sell_markers.loc[i,'price']), "▼", ax=ax, color="#ff00c8", anchor=(0.5,0.5)).setZValue(101)
    for i in buy_markers.index:
        fplt.add_text((buy_markers.loc[i,'entry_time'], buy_markers.loc[i,'price']), "▲", ax=ax, color="#4400ff", anchor=(0.5,0.5)).setZValue(101)

    # for all completed trades (trade frame)
    for i in tf.index:

        _color = "#11cc00" if tf.loc[i, 'pnl'] > 0 else "#cc0000"
        # trade line
        line = fplt.add_line(( tf.loc[i,'entry_time'], tf.loc[i,'entry_price']), (tf.loc[i,'exit_time'], tf.loc[i,'exit_price']),
                                ax = ax, 
                                color=_color, 
                                style="-",
                                width=2).setZValue(100)
        
        
        if trade_id:
            fplt.add_text((tf.loc[i,'entry_time'], tf.loc[i,'entry_price']), f"{tf.loc[i,'trade_id']}", color=_color, ax=ax)

        if boxes:
            # SL and TP lines
            #fplt.add_line((tf.loc[i,'entry_time'], tf.loc[i,'sl']), (tf.loc[i,'exit_time']+ timestep, tf.loc[i,'sl']), ax = ax, color="#ff0000")
            #fplt.add_line((tf.loc[i,'entry_time'], tf.loc[i,'tp']), (tf.loc[i,'exit_time']+ timestep, tf.loc[i,'tp']), ax = ax, color="#15ff00")
            
            # SL and TP areas
            fplt.add_rect((tf.loc[i,'entry_time'], tf.loc[i,'entry_price']), (tf.loc[i,'exit_time']+ timestep, tf.loc[i,'tp']), ax = ax, color="#74e4745f") 
            fplt.add_rect((tf.loc[i,'entry_time'], tf.loc[i,'entry_price']), (tf.loc[i,'exit_time']+ timestep, tf.loc[i,'sl']), ax = ax, color="#e481745f") 

def plot_candles(start , ax):
    
    return

def plot_timeblock(df, zone_col, ax, color="#2448e960", title=""):
    """
    Plot time zones (positive blocks) as rectangles on a finplot chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'high' and 'low' columns and a datetime index.
    zone_col : str
        Column name in df that contains zone IDs (positive/negative ints).
    ax : finplot.FinplotWidget
        Finplot axis handle.
    color : str
        RGBA hex color for the rectangles (default semi-transparent blue).
    title : str
        Optional label text placed at the start of each zone.
    """

    # Keep only positive (active) zones
    df_zones = df[df[zone_col] > 0].copy()

    # Group by zone ID (e.g. 1, 2, 3...)
    for zone_id, group in df_zones.groupby(zone_col):
        start = group.index[0]
        end = group.index[-1]
        low = group["low"].min()
        high = group["high"].max()

        # Draw rectangle covering that timeblock
        fplt.add_rect((start, low),(end, high),ax=ax,color=color,)
        # Optionally label the start of each zone
        if title:
            fplt.add_text((start, low),f"{title} {zone_id}",color=color,ax=ax,)
            
def plot_timeband(df, column, ax, color="#bdbdbd40", title=""):

    df_band = df.dropna(subset=[column])

    for day, group in df_band.groupby(df_band.index.date):
        fplt.add_vertical_band(group.index[0], group.index[-1], color=color, title=title)

def plot_signal(signal: pd.Series, y: pd.Series, ax, color="#0FFF2F", style="o", title="", width = 1):
    """Input signal is a time series of 0–1 or True/False values.
    y is where you want to display the signal mark."""
    
    mask = signal.astype(bool)
    y_mask = y
    y_mask[mask==False] = np.nan
    
    fplt.plot(y,ax=ax,style=style,color=color,width=width, legend=title).setZValue(101)


def basic_graph(x,y,color="#2ecc71",xlabel="X-axis",ylabel="Y-axis", discord=True):

    # generate your plot
    plt.style.use("cyberpunk")
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    ax.plot(x, y, color=color, marker='o')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cyberpunk.add_glow_effects()

    figure = format_graph(fig, [ax], discord)
    return figure



