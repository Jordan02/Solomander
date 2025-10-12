import pandas as pd
import finplot as fplt
import pytz
from datetime import time


try:
    from .logger import log, stamp, pront
    from .data import load_yfinance
except ImportError:
    # if __name__ == "__main__"
    from logger import log, stamp, pront
    from data import load_yfinance
    




def vwap(df:pd.DataFrame, start = "00:00", end = "23:59", mode="session", tz="UTC"): 

    # local df copy
    df_timezone = df.index.tz # save orginal tz format
    df_local = df.tz_convert(tz)

    # get group container info
    if mode == 'session':
        group_mode=df.index.date
        sa,sb = map(int, start.split(':'))
        ea,eb = map(int, end.split(':'))
        START_T = time(sa, sb)
        END_T = time(ea, eb)

    elif mode == 'daily':
        group_mode=df.index.date
        START_T = time(0,0)
        END_T = time(23,59)

    elif mode == 'weekly':
        group_mode=df.index.to_period("W")
        START_T = time(0,0)
        END_T = time(23,59)

    elif mode == 'monthly':
        group_mode=df.index.to_period("M")
        START_T = time(0,0)
        END_T = time(23,59)

    # group calculations
    vwap_series = pd.Series(index=df_local.index, dtype='float64')
    tp = (df_local['high'] + df_local['low'] + df_local['close'])/3

    for day, group in df_local.groupby(group_mode):
        group = group.between_time(START_T, END_T)
        if group.empty:
            continue
        # vwap = cumsum(price * volume) / cumsum(volume)
        vwap = (tp.loc[group.index] * group['volume']).cumsum() / group['volume'].cumsum()
        vwap_series.loc[group.index] = vwap

    # returning and plotting data
    return vwap_series.tz_convert(df_timezone) # revert time

def timeband(df, start="9:30", end="16:00", tz="UTC"):

    """
    Highlights a time band (session) on a finplot chart and adds a boolean mask to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex (must be timezone-aware).
    ax : finplot.Axis, optional
        finplot axis to plot the time band on. If None, only the mask is added to the DataFrame.
    title : str, optional
        Label for the time band, used for the mask column name and plot annotation.
    start : str, optional
        Start time of the band in "HH:MM" format (default "9:30").
    end : str, optional
        End time of the band in "HH:MM" format (default "16:00").
    tz : str, optional
        Timezone for the session (default "UTC").
    color : str, optional
        Color for the band in hex or RGBA format (default "#bdbdbd40").

    Returns
    -------
    None
        Adds a boolean mask column to the DataFrame and optionally plots the band on the chart.
    """

    # extract time
    sa,sb = map(int, start.split(":"))
    ea,eb = map(int,end.split(":"))
    START = time(sa, sb)
    END = time(ea, eb)

    # create timeband
    df_timezone = df.index.tz # save orginal tz format
    df_band = df.tz_convert(tz)
    df_band = df_band.between_time(START, END)

    # add boolean series to df
    timeband_series = pd.Series(index=df_band.index, dtype='bool')
    timeband_series[:] = True
    return timeband_series.tz_convert(df_timezone) # revert time
            
def timeblock(df, start="9:30", end="16:00", tz="UTC"):

    # extracting time
    sa,sb = map(int, start.split(":"))
    ea,eb = map(int,end.split(":"))
    START = time(sa, sb)
    END = time(ea, eb)

    # convert to TZ time and filter between NY hours
    df_timezone = df.index.tz # save orginal tz format
    df_sess = df.tz_convert(tz)
    df_sess = df_sess.between_time(START, END)

    # add boolean series to df
    timeblock_series = pd.Series(index=df_sess.index, dtype='bool')
    timeblock_series[:] = True
    return timeblock_series.tz_convert(df_timezone) # revert time

def sessions(df):

    NY = timeblock(df, start="13:30", end="20:00", tz="UTC")
    LDN = timeblock(df, start="7:30", end="15:30", tz="UTC")
    TKY = timeblock(df, start="00:00", end="06:00", tz="UTC")

    return {"NY": NY, "LDN": LDN, "TKY": TKY}




if __name__ == "__main__":

    from visuals import plot_timeblock, plot_timeband 

    df = load_yfinance("MNQ=F", start="2025-08-16", end="2025-09-16", interval="5m")
    fplt.display_timezone = pytz.timezone("UTC")

    ax, ax2 = fplt.create_plot('MNQ Chart', rows=2)
    fplt.candlestick_ochl(df, ax=ax)

    df['vwap'] = vwap(df, mode="daily")
    df['time_block'] = timeblock(df, start="9:30", end="16:00", tz="America/New_York")
    df['tb_NY'] = sessions(df)['NY']
    df['tb_LDN'] = sessions(df)['LDN']       
    df['tb_TKY'] = sessions(df)['TKY']
    df['tband_rth'] = timeband(df)

    fplt.plot(df['vwap'], ax=ax, color="#219bec", legend="VWAP")
    plot_timeblock(df, 'time_block', ax=ax, color="#2448e960", title="RTH")
    plot_timeblock(df, 'tb_NY', ax=ax, color="#d5e9245f", title="NY")
    plot_timeblock(df, 'tb_LDN', ax=ax, color="#1333c55f", title="LDN")
    plot_timeblock(df, 'tb_TKY', ax=ax, color="#df74af5f", title="TKY")
    plot_timeband(df, 'tband_rth', ax=ax, color="#bdbdbd26", title="RTH")

    pront.info(df.head(100))
    pront.info(df['tband_rth'].sum())
    pront.info(len(df))
   

    fplt.volume_ocv(df[['open', 'close', 'volume']], ax=ax2)
    fplt.show()

