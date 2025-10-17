import pandas as pd
import finplot as fplt
import pytz
from datetime import time


try:
    from .logger import log, stamp, pront
    from .data import yfin_load_data
except ImportError:
    # if __name__ == "__main__"
    from logger import log, stamp, pront
    from data import yfin_load_data
    




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

            
def timeblock(df, start="9:30", end="16:00", tz="UTC"):

    # extracting time
    sa,sb = map(int, start.split(":"))
    ea,eb = map(int,end.split(":"))
    START = time(sa, sb)
    END = time(ea, eb)

    # convert to TZ time and filter between NY hours
    df_timezone = df.index.tz # save orginal tz format
    df_sess = df.tz_convert(tz)

    # Boolean mask of whether each timestamp is inside the defined time window
    in_zone = pd.Series(False, index=df_sess.index)
    in_zone.iloc[df_sess.index.indexer_between_time(START, END)] = True

    # Detect transitions between True/False
    transitions = in_zone.ne(in_zone.shift()).cumsum()

    # Assign alternating positive/negative IDs
    zone_ids = (transitions + 1) // 2  # start counting at 1
    zone_ids = zone_ids.where(in_zone, -zone_ids)

    return zone_ids.tz_convert(df_timezone)

def timeblock_value(df, value='close', agg='max', start="9:30", end="16:00", tz="UTC", mode='in_zones'):
    """
    Returns a Series with aggregated values of `df[value]` for each time block.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DateTimeIndex (tz-aware) and the target column.
    value : str
        Column name to aggregate (e.g. 'close', 'high', etc.)
    agg : str
        Aggregation function: 'max', 'min', 'mean', or 'median'.
    start, end : str
        Time window defining the active zone (e.g. "09:30", "16:00").
    tz : str
        Timezone used for defining the time window (converted temporarily).
    mode : str, default 'in_zones'
        - 'in_zones': fill only active zones (NaN outside)
        - 'all_zones': fill both active and inactive zones separately
        - 'extend': fill active zones and extend their value forward until next reset

    Returns
    -------
    pd.Series
        Same index as df (in original timezone).
        Aggregated values repeated according to mode.
    """
    # --- extract times ---
    sa, sb = map(int, start.split(":"))
    ea, eb = map(int, end.split(":"))
    START, END = time(sa, sb), time(ea, eb)

    # --- preserve original tz ---
    orig_tz = df.index.tz
    df_sess = df.tz_convert(tz)

    # --- boolean mask for in-zone timestamps ---
    in_zone = pd.Series(False, index=df_sess.index)
    in_zone.iloc[df_sess.index.indexer_between_time(START, END)] = True

    # --- compute alternating zone IDs ---
    transitions = in_zone.ne(in_zone.shift()).cumsum()
    zone_ids = (transitions + 1) // 2
    zone_ids = zone_ids.where(in_zone, -zone_ids)

    # --- compute aggregate per zone ---
    result = pd.Series(index=df_sess.index, dtype=float)

    for zone_id, group in df_sess.groupby(zone_ids):
        vals = group[value]
        if agg == 'max':
            v = vals.max()
        elif agg == 'min':
            v = vals.min()
        elif agg in ('mean', 'avg', 'average'):
            v = vals.mean()
        elif agg == 'median':
            v = vals.median()
        else:
            raise ValueError(f"Unknown agg type '{agg}'")

        if mode == 'all_zones' or (mode == 'in_zones' and zone_id > 0):
            result.loc[group.index] = v
        elif mode == 'extend' and zone_id > 0:
            result.loc[group.index] = v
        else:
            result.loc[group.index] = pd.NA

    # --- handle extend mode ---
    if mode == 'extend':
        result = result.ffill()

    # --- convert back to original timezone ---
    result = result.tz_convert(orig_tz)
    return result
    

def sessions(df):

    NY = timeblock(df, start="13:30", end="20:00", tz="UTC")
    LDN = timeblock(df, start="7:30", end="15:30", tz="UTC")
    TKY = timeblock(df, start="00:00", end="06:00", tz="UTC")

    return {"NY": NY, "LDN": LDN, "TKY": TKY}


def sessions_value(df, value='close', agg='max', mode="extend"):
    """
    Compute an aggregated value of a chosen column within repeating time-based zones.

    For each time window (e.g., a market session), the function calculates an 
    aggregate statistic (max, min, mean, median) of the specified column, then 
    fills the result according to the selected `mode`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a timezone-aware DateTimeIndex and the target column.
    value : str, default 'close'
        Column name to aggregate (e.g., 'close', 'high', 'low', 'volume', etc.).
    agg : str, default 'max'
        Aggregation function to apply within each zone.
        Supported options: 'max', 'min', 'mean', 'median'.
    start : str, default '09:30'
        Start time of the active zone (HH:MM format).
    end : str, default '16:00'
        End time of the active zone (HH:MM format).
    tz : str, default 'UTC'
        Timezone in which the `start` and `end` times are defined.
        The index is temporarily converted to this timezone for accurate filtering,
        then restored to the DataFrame’s original timezone.
    mode : {'in_zones', 'all_zones', 'extend'}, default 'in_zones'
        Defines how the aggregated values are applied across the timeline:
        
        - 'in_zones' : Fill only within active time windows; NaN elsewhere.
        - 'all_zones' : Fill both in-zone and out-of-zone segments with 
          their respective aggregated values.
        - 'extend' : Fill active zones normally, then forward-fill their value 
          until the next active zone begins.

    Returns
    -------
    pd.Series
        Series aligned to `df.index` (in original timezone), containing the 
        aggregated values repeated across time according to the selected `mode`.
    """

    NY  = timeblock_value(df, value=value, agg=agg, start="13:30", end="20:00", tz="UTC", mode=mode)
    LDN = timeblock_value(df, value=value, agg=agg, start="7:30",  end="15:30", tz="UTC", mode=mode)
    TKY = timeblock_value(df, value=value, agg=agg, start="00:00", end="06:00", tz="UTC", mode=mode)

    return {"NY": NY, "LDN": LDN, "TKY": TKY}






if __name__ == "__main__":

    from visuals import plot_timeblock, plot_timeband 

    df = yfin_load_data("MNQ=F", start="2025-08-16", end="2025-09-16", interval="5m")
    fplt.display_timezone = pytz.timezone("UTC")

    ax, ax2 = fplt.create_plot('MNQ Chart', rows=2)
    fplt.candlestick_ochl(df, ax=ax)

    df['vwap'] = vwap(df, mode="daily")
    df['time_block'] = timeblock(df, start="9:30", end="16:00", tz="America/New_York")
    df['tb_NY'] = sessions(df)['NY']
    df['tb_LDN'] = sessions(df)['LDN']       
    df['tb_TKY'] = sessions(df)['TKY']

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

