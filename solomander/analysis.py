import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm, skew


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



def monte_carlo (tf:pd.DataFrame, runs:int=100, mode:str="permutation"):
    
    # permuation Monte Carlo = no resampling, just shuffle the trades
    # Bootsrap Monte Carlo = resample with replacement
    fig = plt.figure(figsize=(10,5))
    gs = fig.add_gridspec(2,2, width_ratios=[3,1])

    # plots
    ax = fig.add_subplot(gs[:,0])
    ax.axhline(0, color="#130000", linestyle="--")
    ax.set_title("Monte Carlo")
    ax.set_xlabel("Trades")
    ax.set_ylabel("PnL")
    ax.legend()

    ax_hist = fig.add_subplot(gs[0,1])
    ax_hist.set_xlabel("Cash")
    ax_hist.set_ylabel("Frequency")

    trades_pnl = tf['pnl'].to_numpy().copy()
    original_mdd = max_drawdown(trades_pnl)
    ax.plot(trades_pnl.cumsum(), color="#0004ff")
    MDD = []


    if mode == "permutation":
        for i in range(runs):
            np.random.shuffle(trades_pnl)
            cum_pnl = np.concatenate([[0], np.cumsum(trades_pnl)])
            MDD.append(max_drawdown(trades_pnl))
            ax.plot(cum_pnl, color="#1900ff21")

    elif mode == "bootstrap":
        
        for i in range(runs):
            sample = np.random.choice(trades_pnl, size=len(trades_pnl), replace=True, p=None)
            cum_pnl = np.concatenate([[0], np.cumsum(sample)])
            MDD.append(max_drawdown(sample))
            ax.plot(cum_pnl, color="#1900ff21")

    # MDD Histogram Calculation
    
    HIST_QTY = 100
    counts, bins = np.histogram(MDD, bins=HIST_QTY)
    shape, loc, scale = skewnorm.fit(MDD)
    xs = np.linspace(min(MDD), 0, runs)
    pdf = skewnorm.pdf(xs, shape, loc, scale) # scale to histogram
    pdf = pdf * (counts.max()/pdf.max())   # normalize
    alpha = 0.05 # location for lower 95% 
    lower_95 = skewnorm.ppf(alpha, shape, loc, scale)
    skew_value = skew(MDD)
    skew_mean = skewnorm.mean(shape, loc, scale)

    # SHOULD YOU WANT confidence interval (CI) on your lower_95 VALUE
    # SE (standard error) = sqrt(q(1-q)/n) where q = 0.05 (lower 95% CI), n = runs
    # we get a z score for this, multiple by SE to get coffset for CI.
    #ci_lower = lower_95 - z * se
    #ci_upper = lower_95 + z * se
    
    
    # Plotting 

    ax_hist.axvline(skew_mean, color="blue", linestyle="--", label="mean: {:.2f}".format(skew_mean))
    ax_hist.axvline(lower_95, color="red", linestyle="--", label="5% VaR: {:.2f}".format(lower_95))
    ax_hist.hist(MDD, bins=HIST_QTY, color="steelblue", edgecolor="black", alpha=0.7)
    ax_hist.plot(xs, pdf, color="red", linewidth=2)
    ax_hist.set_title("MDD Histogram")

    textstr = '\n'.join((
    f"Skew: {skew_value:.2f}",
    f"Skewed Mean: {skew_mean:.1f}",
    f"95% Lower CL: {lower_95:.1f}",
    f"Original MDD: {original_mdd:.2f}"
    ))


    ax_hist.text(lower_95, ax_hist.get_ylim()[1]*0.5, f"{lower_95:.1f}", 
        color="red", rotation=90, va="center", ha="right", fontsize=9)

    ax_hist.text(0.05, 0.95, textstr,
                    transform=ax_hist.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    horizontalalignment='left',
                    rotation_mode='anchor',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7))

    textstrmain = '\n'.join((
    f"tests: {runs}",
    f"Mode: {mode}"
    ))

    ax.text(0.05, 0.95, textstrmain,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    horizontalalignment='left',
                    rotation_mode='anchor',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7))

    

    plt.show()

    return