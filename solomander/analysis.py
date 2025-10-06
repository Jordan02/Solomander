import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm, skew

try:
    from .logger import log, stamp, pront
    from .baseStrategy import Strategy
    from .utils import random_color, max_drawdown, sharpe, sortino
except ImportError:
    from logger import log, stamp, pront
    from solomander.baseStrategy import Strategy
    from .utils import random_color, max_drawdown, sharpe, sortino
  




def monte_carlo (tf:pd.DataFrame, runs:int=100, seed:int=None, mode:str="bootstrap", params=None):
    
    # CHECK IF TRADE DF IS EMPTY
    if tf.empty == True:
            log.warning("No orders were executed. Check your strategy logic.")
            return

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

    if params is not None:
        text_params = '\n'.join([f"{k}: {v:<.2f}" for k, v in params.items()])
        text_data = '\n'.join([text_data, text_params])

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

    _plot_histogram(ax_hist, MDD, bin_qty=50, textstr=f"Original MDD: {original_mdd:.2f}", title="Drawdown", xlabel="")
    if mode == "bootstrap":
        # These distrubations only occur for bootstrap mode
        ax_pnl = fig.add_subplot(gs[2,1])
        ax_sr = fig.add_subplot(gs[1,1])
        _plot_histogram(ax_sr, SR, bin_qty=50, textstr=f"Original SR: {original_sr:.2f}", title="Sharpe Ratio", xlabel="")
        _plot_histogram(ax_pnl, PNL, bin_qty=50, textstr=f"Original PnL: {original_pnl:.2f}", title="PnL", xlabel="")

    ax.plot(orginal_trades.cumsum(), color="#ff00f2") # orginal

    plt.show()

    return


def monte_carlo_metric(tf:pd.DataFrame,runs:int=100,seed:int=42, mode:str="bootstrap"):

    # CHECK IF TRADE DF IS EMPTY
    if tf.empty == True:
            log.warning("No orders were executed. Check your strategy logic.")
            return
    
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


def _plot_histogram(ax, data, bin_qty=50, textstr="", title="Histogram", xlabel="Value"):

    # histogram Calculation
    counts, bins = np.histogram(data, bins=bin_qty)
    shape, loc, scale = skewnorm.fit(data)
    xs = np.linspace(min(data), max(data), bin_qty)
    pdf = skewnorm.pdf(xs, shape, loc, scale) # scale to histogram
    pdf = pdf * (counts.max()/pdf.max())   # normalize
    alpha = 0.05 # location for lower 95% 
    lower_95 = skewnorm.ppf(alpha, shape, loc, scale)
    upper_95 = skewnorm.ppf(1-alpha, shape, loc, scale)
    skew_value = skew(data)
    skew_mean = skewnorm.mean(shape, loc, scale)

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
    ax.annotate(f"  {upper_95:.1f}", 
                    xy=(upper_95, ax.get_ylim()[1]*0.5),
                    xytext=(5,0), textcoords="offset points",  # fixed 5pt offset
                    color="red", rotation=90,
                    va="center", ha="left", fontsize=8)

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
                        va='top',
                        ha='left',
                        rotation_mode='anchor',
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7))

    pass


def noise_test(strategy: Strategy, test_params: dict, nudges:int=3):
    
    # CHECK IF TRADE DF IS EMPTY
    if strategy.tf.empty == True:
            log.warning("No orders were executed. Check your strategy logic.")
            return

    tf = strategy.tf.copy()
    pnls = []
    pnl_final = []
    srs = []

    for key, offset  in test_params.items():

        if key not in strategy.INPUT_PARAMS:
            log.error(f"Parameter '{key}' not found in strategy INPUT_PARAMS")
            return
        
        val = getattr(strategy, key) # value of original param

        for i in range(-nudges, nudges+1):

            if i==0:
                continue # skip original value
            
            nudged_param = val + offset * i
            new_kwargs = strategy.init_kwargs.copy()
            new_kwargs[key] = nudged_param
            strat_nudged = strategy.__class__(df=strategy.df, **new_kwargs)
            strat_nudged.execute()

            pnls.append((key,nudged_param, strat_nudged.df_cum_pnl.to_numpy()))
            pnl_final.append(strat_nudged.PNL)
            srs.append(strat_nudged.SHARPE_RATIO_ANNUAL)

    
    fig = plt.figure(figsize=(10,5))
    gs = fig.add_gridspec(2,2, width_ratios=[2,1])
    ax = fig.add_subplot(gs[:,0])
    ax_sr = fig.add_subplot(gs[0,1])
    ax_pnl = fig.add_subplot(gs[1,1])

    ax.axhline(0, color="#130000", linestyle="--")
    ax.set_title("Noise Test")
    ax.set_xlabel("Trades") 
    ax.set_ylabel("PnL")


    # ==== ORIGINAL CURVE ====
    x = np.arange(len(strategy.df_cum_pnl))
    y = strategy.df_cum_pnl.to_numpy()
    ax.plot(x,y, color='red', label="Original")
    param_str = ", ".join(  f"{p}={strategy.init_kwargs[p]:<.2f}" 
                            for p in test_params.keys() 
                            if p in strategy.init_kwargs
                        )

    ax.annotate(    param_str,
                    xy=(x[-1], y[-1]),
                    xytext=(3,0),  # offset to the right
                    textcoords="offset points",
                    fontsize=6,
                    color='red'
                )
    
    # ==== NUDGED CURVES ====
    color = random_color(alpha=0.5)
    switch_color = 0

    for param,nudged_param,curve in pnls:
        
        if switch_color >= nudges*2: 
            color = random_color(alpha=0.5)
            switch_color = -1
        switch_color +=1

        x = np.arange(len(curve))
        ax.plot(x, curve, color=color)

        ax.annotate(    f"{param}={nudged_param:<.2f}",
                        xy=(x[-1], curve[-1]),
                        xytext=(3,0),  # offset to the right
                        textcoords="offset points",
                        fontsize=6,
                        color=color
                    )
    
    # ==== Histogram ====
    _plot_histogram(ax_sr, srs, bin_qty=50, title="", xlabel="SR", textstr=f"Original SR: {strategy.SHARPE_RATIO_ANNUAL:.2f}")
    _plot_histogram(ax_pnl, pnl_final, bin_qty=50, title="", xlabel="PnL", textstr=f"Original PnL: {strategy.PNL:.2f}")

    plt.show()


        # === get attribute type ===



    return