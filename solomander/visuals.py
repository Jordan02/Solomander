import finplot as fplt
import pandas as pd
import pyqtgraph as pg


try:

    from .logger import log, stamp, pront
    from .data import load_yfinance
    
except ImportError:
    
    from logger import log, stamp, pront
    from data import load_yfinance
    
    


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

def plot_timeblock(df, column, ax, color="#2448e960", title=""):



    df_sess = df.dropna(subset=[column])

    for day, group in df_sess.groupby(df_sess.index.date):
        fplt.add_rect((group.index[0], group['low'].min()), (group.index[-1], group['high'].max()), ax=ax, color=color)
        fplt.add_text((group.index[0], group['low'].min()), title, color=color, ax=ax)

def plot_timeband(df, column, ax, color="#bdbdbd40", title=""):

    df_band = df.dropna(subset=[column])

    for day, group in df_band.groupby(df_band.index.date):
        fplt.add_vertical_band(group.index[0], group.index[-1], color=color)