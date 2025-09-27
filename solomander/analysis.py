import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm, skew

try:
    from .logger import log, stamp, pront
except ImportError:
    from logger import log, stamp, pront
  


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



def monte_carlo (tf:pd.DataFrame, runs:int=100, seed:int=None, mode:str="bootstrap"):
    
    # permuation Monte Carlo = no resampling, just shuffle the trades
    # Bootsrap Monte Carlo = resample with replacement
    
    if seed is not None:
        np.random.seed(seed)
    else:
        seed = np.random.randint(0, 1000000)

    trades_pnl_pd = tf[['exit_time','pnl','return']]
    orginal_trades = trades_pnl_pd["pnl"].to_numpy()
    original_mdd = max_drawdown(trades_pnl_pd['pnl'])
    original_pnl = orginal_trades.sum()
    original_sr = sharpe(trades_pnl_pd, type="annual")
    MDD = []
    PNL = []
    SR = []

    fig = plt.figure(figsize=(10,5))
    gs = fig.add_gridspec(3,2, width_ratios=[2,1])
    ax = fig.add_subplot(gs[:,0])
    ax_hist = fig.add_subplot(gs[0,1])

    
    # plots
    ax.axhline(0, color="#130000", linestyle="--")
    ax.set_title("Monte Carlo")
    ax.set_xlabel("Trades")
    ax.set_ylabel("PnL")
    ax.legend()
    
    text_data = '\n'.join((
            f"runs: {runs}",
            f"resampling mode: {mode}",
            f"seed: {seed}"
            ))

    ax.text(0.05, 0.95, text_data,
                    transform=ax.transAxes,
                    fontsize=7,
                    verticalalignment='top',
                    horizontalalignment='left',
                    rotation_mode='anchor',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7))
    
    if mode == "permutation":
        
        for i in range(runs):
            
            sample_pd = trades_pnl_pd.sample(frac=1, replace=False, random_state=seed+i)
            sample_pnl = sample_pd['pnl'].to_numpy()
            sumcum_pnl = sample_pnl.cumsum()

            PNL.append(sumcum_pnl[-1])
            MDD.append(max_drawdown(sample_pnl))
            SR.append(sharpe(sample_pd, type="annual")) # assuming daily trades
            
            ax.plot(sumcum_pnl, color="#1900ff1a")
            

    elif mode == "bootstrap":

        for i in range(runs):
           
            sample_pd = trades_pnl_pd.sample(frac=1, replace=True, random_state=seed+i)
            sample_pnl = sample_pd['pnl'].to_numpy()
            sumcum_pnl = sample_pnl.cumsum()

            PNL.append(sumcum_pnl[-1])
            MDD.append(max_drawdown(sample_pnl))
            SR.append(sharpe(sample_pd, type="annual")) # assuming daily trades
            
            ax.plot(sumcum_pnl, color="#1900ff1a")

    # MDD Histogram Calculation

    _plot_histogram(ax_hist, MDD, runs, bin_qty=50, textstr=f"Original MDD: {original_mdd:.2f}", title="Drawdown", xlabel="")
    if mode == "bootstrap":
        # These distrubations only occur for bootstrap mode
        ax_pnl = fig.add_subplot(gs[2,1])
        ax_sr = fig.add_subplot(gs[1,1])
        _plot_histogram(ax_sr, SR, runs, bin_qty=50, textstr=f"Original SR: {original_sr:.2f}", title="Sharpe Ratio", xlabel="")
        _plot_histogram(ax_pnl, PNL, runs, bin_qty=50, textstr=f"Original PnL: {original_pnl:.2f}", title="PnL", xlabel="")

    ax.plot(orginal_trades.cumsum(), color="#ff00f2") # orginal

    plt.show()

    return


def monte_carlo_metric(tf:pd.DataFrame,runs:int=100,seed:int=42, mode:str="bootstrap"):

    # permuation Monte Carlo = no resampling, just shuffle the trades
    # Bootsrap Monte Carlo = resample with replacement
    
    if seed is not None:
        np.random.seed(seed)
    else:
        seed = np.random.randint(0, 1000000)

    trades_pnl_pd = tf[['exit_time','pnl','return']]
    #orginal_trades = trades_pnl_pd["pnl"].to_numpy()
    MDD = []
    PNL = []
    SR = []

    if mode == "permutation":
        
        for i in range(runs):
            

            sample_pd = trades_pnl_pd.sample(frac=1, replace=False, random_state=seed+i)
            sample_pnl = sample_pd['pnl'].to_numpy()
            sumcum_pnl = sample_pnl.cumsum()

            PNL.append(sumcum_pnl[-1])
            MDD.append(max_drawdown(sample_pnl))
            SR.append(sharpe(sample_pd, type="annual")) # assuming daily trades
            
    elif mode == "bootstrap":

        for i in range(runs):
           
            sample_pd = trades_pnl_pd.sample(frac=1, replace=True, random_state=seed+i)
            sample_pnl = sample_pd['pnl'].to_numpy()
            sumcum_pnl = sample_pnl.cumsum()

            PNL.append(sumcum_pnl[-1])
            MDD.append(max_drawdown(sample_pnl))
            SR.append(sharpe(sample_pd, type="annual")) # assuming daily trades

    return {
        "pnl": np.array(PNL),
        "mdd": np.array(MDD),
        "sr": np.array(SR)
    }

    



def _plot_histogram(ax, data, runs, bin_qty=100, textstr="", title="Histogram", xlabel="Value"):

    # histogram Calculation
    counts, bins = np.histogram(data, bins=bin_qty)
    shape, loc, scale = skewnorm.fit(data)
    xs = np.linspace(min(data), max(data), runs)
    pdf = skewnorm.pdf(xs, shape, loc, scale) # scale to histogram
    pdf = pdf * (counts.max()/pdf.max())   # normalize
    alpha = 0.05 # location for lower 95% 
    lower_95 = skewnorm.ppf(alpha, shape, loc, scale)
    upper_95 = skewnorm.ppf(1-alpha, shape, loc, scale)
    skew_value = skew(data)
    skew_mean = skewnorm.mean(shape, loc, scale)

    # SHOULD YOU WANT confidence interval (CI) on your lower_95 VALUE
    # SE (standard error) = sqrt(q(1-q)/n) where q = 0.05 (lower 95% CI), n = runs
    # we get a z score for this, multiple by SE to get coffset for CI.
    #ci_lower = lower_95 - z * se
    #ci_upper = lower_95 + z * se
    
    # labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)

    # Plotting histogram
    ax.axvline(skew_mean, color="blue", linestyle="--", label="mean: {:.2f}".format(skew_mean))
    ax.hist(data, bins=bin_qty, color="steelblue", edgecolor="black", alpha=0.7)
    ax.plot(xs, pdf, color="red", linewidth=2)

    #lower 95% limit
    ax.axvline(lower_95, color="red", linestyle="--", label="5% VaR: {:.2f}".format(lower_95))
    ax.text(lower_95, ax.get_ylim()[1]*0.5, f"{lower_95:.1f}", 
        color="red", rotation=90, va="center", ha="right", fontsize=9)
    
    #upper 95% limit
    xlim = ax.get_xlim()
    ax.axvline(upper_95, color="red", linestyle="--", label="95% VaR: {:.2f}".format(upper_95))
    ax.text(upper_95 + (xlim[1]-xlim[0])*0.07, ax.get_ylim()[1]*0.5, f"{upper_95:.1f}", 
        color="red", rotation=90, va="center", ha="right", fontsize=9)

    # text box
    text_data = '\n'.join((
            f"Skew: {skew_value:.2f}",
            f"stddev: {np.std(data, ddof=1):.1f}",
            f"Skewed Mean: {skew_mean:.1f}",
            f"95% Lower CL: {lower_95:.1f}",
            f"95% Upper CL: {upper_95:.1f}"
            ))
    
    textstr = '\n'.join([text_data, textstr])

    ax.text(0.05, 0.95, textstr,
                    transform=ax.transAxes,
                    fontsize=7,
                    verticalalignment='top',
                    horizontalalignment='left',
                    rotation_mode='anchor',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7))

    pass

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


