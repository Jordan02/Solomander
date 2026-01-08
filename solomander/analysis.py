import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm, skew
import mplcyberpunk as cyberpunk
import io
import statsmodels.api as sm

try:
    from .logger import log, stamp, pront
    from .baseStrategy import Strategy
    from .utils import random_color, max_drawdown, sharpe, sortino, adjust_opacity, shift_hue, format_graph, load_graph_color
    from .backtester import Backtester
except ImportError:
    from logger import log, stamp, pront
    from solomander.baseStrategy import Strategy
    from .utils import random_color, max_drawdown, sharpe, sortino, adjust_opacity, shift_hue, format_graph, load_graph_color
    from backtester import Backtester
  




def monte_carlo (results:Strategy, runs:int=100, seed:int=None, discord = False, mode:str="bootstrap", params=None, color:str="#2600ff"):
    
    # CHECK IF TRADE DF IS EMPTY
    if results.tf.empty == True:
            log.warning("No orders were executed. Check your strategy logic.")
            return

    # permutation Monte Carlo = no resampling, just shuffle the trades
    # Bootstrap Monte Carlo = resample with replacement

    if seed is not None:
        np.random.seed(seed)
    else:
        seed = np.random.randint(0, 1000000)

    trades_pnl_pd = results.tf[['exit_time','pnl','return']]
    orginal_pnls = results.l_CUMSUM_PNL.copy()

    original_mdd_total = results.TOTAL_MAX_DRAWDOWN
    original_pnl_total = results.TOTAL_PNL
    original_sr_total = results.TOTAL_SHARPE_RATIO_ANNUAL
    curve_color = adjust_opacity(color, 0.1)

    MDD = []
    PNL = []
    SR = []

    plt.style.use("cyberpunk")
    fig = plt.figure(figsize=(14,6))
    gs = fig.add_gridspec(3,2, width_ratios=[2,1])
    ax = fig.add_subplot(gs[:,0])
    ax_hist = fig.add_subplot(gs[0,1])

    ax.plot(orginal_pnls, color=color) # orginal
    cyberpunk.make_lines_glow(ax)

    # plots
    graph_colors = load_graph_color()
    ax.axhline(0, color=graph_colors["grid"], linestyle="--")
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

    ax.text(0.05, 0.95, text_data,
                    transform=ax.transAxes,
                    fontsize=7,
                    verticalalignment='top',
                    horizontalalignment='left',
                    rotation_mode='anchor',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor=graph_colors["grid"], facecolor="none", alpha=0.7))
    
    if mode == "permutation":
        
        for i in range(runs):
            
            sample_pd = trades_pnl_pd.sample(frac=1, replace=False, random_state=seed+i)
            
            sample_pnl = sample_pd['pnl'].to_numpy()
            sumcum_pnl = sample_pnl.cumsum()
            sample_returns = sample_pd['return'].to_numpy() 
            sample_dates = sample_pd['exit_time']

            PNL.append(sumcum_pnl[-1])
            MDD.append(max_drawdown(sample_pnl))
            SR.append(sharpe(sample_returns, sample_dates, mode="annual")) # assuming daily trades
            
            ax.plot(sumcum_pnl, color=curve_color, label="_nolegend_")
            

    elif mode == "bootstrap":

        for i in range(runs):
           
            sample_pd = trades_pnl_pd.sample(frac=1, replace=True, random_state=seed+i)

            sample_pnl = sample_pd['pnl'].to_numpy()
            sumcum_pnl = sample_pnl.cumsum()
            sample_returns = sample_pd['return'].to_numpy() 
            sample_dates = sample_pd['exit_time']

            PNL.append(sumcum_pnl[-1])
            MDD.append(max_drawdown(sample_pnl))
            SR.append(sharpe(sample_returns, sample_dates, mode="annual")) # assuming daily trades
            
            ax.plot(sumcum_pnl, color=curve_color, label="_nolegend_")

    # MDD Histogram Calculation

    _plot_histogram(ax_hist, MDD, bin_qty=50, textstr=f"Original MDD: {original_mdd_total:.2f}", title="", xlabel="Drawdown", color=color)
    if mode == "bootstrap":
        # These distrubations only occur for bootstrap mode
        ax_pnl = fig.add_subplot(gs[2,1])
        ax_sr = fig.add_subplot(gs[1,1])
        _plot_histogram(ax_sr, SR, bin_qty=50, textstr=f"Original SR: {original_sr_total:.2f}", title="", xlabel="Sharpe Ratio", color=color)
        _plot_histogram(ax_pnl, PNL, bin_qty=50, textstr=f"Original PnL: {original_pnl_total:.2f}", title="", xlabel="PnL", color=color)

        return format_graph(fig, [ax, ax_hist, ax_pnl, ax_sr], discord)

    else:

        return format_graph(fig, [ax, ax_hist], discord)    

    


   
def monte_carlo_metric(results:Strategy, runs:int=100,seed:int=42, mode:str="bootstrap"):

    # CHECK IF TRADE DF IS EMPTY
    if results.tf.empty == True:
            log.warning("No orders were executed. Check your strategy logic.")
            return
    
    # permuation Monte Carlo = no resampling, just shuffle the trades
    # Bootsrap Monte Carlo = resample with replacement
    
    if seed is not None:
        np.random.seed(seed)
    else:
        seed = np.random.randint(0, 1000000)

    trades_pnl_pd = results.tf[['exit_time','pnl','return']]
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
            SR.append(sharpe(sample_pd, mode="annual")) # assuming daily trades
            
    elif mode == "bootstrap":

        for i in range(runs):
           
            sample_pd = trades_pnl_pd.sample(frac=1, replace=True, random_state=seed+i)
            sample_pnl = sample_pd['pnl'].to_numpy()
            sumcum_pnl = sample_pnl.cumsum()

            PNL.append(sumcum_pnl[-1])
            MDD.append(max_drawdown(sample_pnl))
            SR.append(sharpe(sample_pd, mode="annual")) # assuming daily trades

    return {
        "pnl": np.array(PNL),
        "mdd": np.array(MDD),
        "sr": np.array(SR)
    }


def _plot_histogram(ax, data, bin_qty=50, textstr="", title="Histogram", xlabel="Value", color="#ff2600", discord = False):

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

    color_adjusted = adjust_opacity(color, 0.3)

    # Plotting histogram
    ax.plot(xs, pdf, color=color, linewidth=2)
    cyberpunk.make_lines_glow(ax)
    graph_color = load_graph_color()
    ax.axvline(skew_mean, color=color, linestyle="--", label="mean: {:.2f}".format(skew_mean))
    ax.hist(data, bins=bin_qty, color=color, edgecolor=graph_color["grid"], alpha=0.3)
    

    #lower 95% limit
    ax.axvline(lower_95, color=color, linestyle="--", label="5% VaR: {:.2f}".format(lower_95))
    ax.text(lower_95, ax.get_ylim()[1]*0.5, f"{lower_95:.1f}", 
        color=color, rotation=90, va="center", ha="right", fontsize=9)
    
    #upper 95% limit
    xlim = ax.get_xlim()
    ax.axvline(upper_95, color=color, linestyle="--", label="95% VaR: {:.2f}".format(upper_95))
    ax.annotate(f"  {upper_95:.1f}", 
                    xy=(upper_95, ax.get_ylim()[1]*0.5),
                    xytext=(5,0), textcoords="offset points",  # fixed 5pt offset
                    color=color, rotation=90,
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

    ax.text(1.02, 0.95, textstr,
                        transform=ax.transAxes,
                        fontsize=7,
                        va='top',
                        ha='left',
                        color=graph_color["text"],
                        rotation_mode='anchor',
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor=graph_color["grid"], facecolor="none", alpha=0.7))


def noise_test(strategy: Strategy, test_params: dict, nudges:int=3, color:str="#ff00c8", discord = False):
    
    # CHECK IF TRADE DF IS EMPTY
    if strategy.symbol_data == {}:
        log.error("❌ Strategy has no data. Make sure to pass the executed result from a backtester babe.")
        return
    
    if strategy.df is None:
        log.error("❌ Strategy has no data. Make sure to pass the executed result from a backtester babe.")
        return
    
    if strategy.tf.empty == True:
            log.warning("❌ No orders were executed. Check your strategy logic.")
            return
    
    df = strategy.df
    ticker = strategy.symbol_data

    pnls = []
    pnl_final = []
    srs = []

    for key, offset  in test_params.items():

        if key not in strategy.INPUT_PARAMS:
            log.error(f"❌ Parameter '{key}' not found in strategy INPUT_PARAMS")
            return
        
        val = getattr(strategy, key) # value of original param

        for i in range(-nudges, nudges+1):

            if i==0:
                continue # skip original value
            
            nudged_param = val + offset * i
            new_kwargs = strategy.INITIAL_KWARGS.copy()
            new_kwargs[key] = nudged_param
            strat_nudged = strategy.__class__(df=strategy.df, **new_kwargs)

            new_study = Backtester(strat_nudged, df,ticker)
            new_results = new_study.execute()
           

            pnls.append((key,nudged_param, new_results.l_CUMSUM_PNL))
            pnl_final.append(new_results.TOTAL_PNL)
            srs.append(new_results.TOTAL_SHARPE_RATIO_ANNUAL)

    plt.style.use("cyberpunk")
    fig = plt.figure(figsize=(14,6))
    gs = fig.add_gridspec(2,2, width_ratios=[2,1])
    ax = fig.add_subplot(gs[:,0])
    ax_sr = fig.add_subplot(gs[0,1])
    ax_pnl = fig.add_subplot(gs[1,1])

    ax.set_title("")
    ax.set_xlabel("Trades") 
    ax.set_ylabel("PnL")

    # ==== ORIGINAL CURVE ====
    orginal_x = np.arange(len(strategy.l_CUMSUM_PNL))
    orginal_y = strategy.l_CUMSUM_PNL
    ax.plot(orginal_x, orginal_y, color=color, label="Original", linewidth=2)
    param_str = "\n".join(  f"{p}={strategy.INITIAL_KWARGS[p]:<.2f}" 
                            for p in test_params.keys() 
                            if p in strategy.INITIAL_KWARGS
                        )
    cyberpunk.make_lines_glow(ax)
    ax.axhline(0, color="white", linestyle="--", alpha=0.2 )

    
    text_obj = ax.text(
                        0.04, 0.98, param_str,
                        transform = ax.transAxes,
                        fontsize=9,
                        va='top',
                        ha='left',
                        rotation_mode='anchor',
                        color=color,
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor="none", alpha=0.9),
                        clip_on=False
)
    
    # ==== NUDGED CURVES ====
    new_color = shift_hue(color, 60)
    new_color_light = adjust_opacity(new_color, 0.2)
    switch_color = 0

    for param,nudged_param,curve in pnls:
        
        if switch_color >= nudges*2: 
            new_color = shift_hue(new_color, 40)
            new_color_light = shift_hue(new_color_light, 40)
            switch_color = -1
        switch_color +=1

        x = np.arange(len(curve))
        ax.plot(x, curve, color=new_color_light)

        ax.annotate(    f"{param}={nudged_param:<.2f}",
                    xy=(x[-1], curve[-1]),
                    xycoords="data",
                    xytext=(1.02,curve[-1]),  # offset to the right
                    textcoords=("axes fraction","data"),
                    fontsize=7,
                    color=new_color,
                    clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor=new_color, facecolor = "none", alpha=0.7, lw=0.5),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color=new_color, alpha=0.3, lw=0.5))


    ax.plot(orginal_x, orginal_y, color=color, label="Original_2", linewidth=2)
    
    # ==== Histogram ====
    _plot_histogram(ax_sr, srs, bin_qty=50, title="", xlabel="SR", textstr=f"Original SR: {strategy.TOTAL_SHARPE_RATIO_ANNUAL:.2f}", color=color)
    _plot_histogram(ax_pnl, pnl_final, bin_qty=50, title="", xlabel="PnL", textstr=f"Original PnL: {strategy.TOTAL_PNL:.2f}", color=color)


    return format_graph(fig, [ax, ax_sr, ax_pnl], discord)




def alpha(strategy: Strategy, discord=False, color = "#dd7600"):

    index_returns = strategy.tf["market_change"]
    strategy_returns = strategy.tf["return"]

    X = sm.add_constant(index_returns)
    y = strategy_returns
    model = sm.OLS(y, X).fit()

    print(model.summary())

    alpha = model.params['const']
    beta = model.params['market_change']
    r2 = model.rsquared

    print(f"\nAlpha: {alpha:.6f}  |  Beta: {beta:.3f}")
    
    plt.style.use("cyberpunk")
    fig, ax = plt.subplots(figsize=(14,6))
    
    plt.scatter(index_returns, strategy_returns, alpha=0.3)
    plt.xlabel("Market Returns")
    plt.ylabel("Strategy Returns")
    plt.title("Alpha/Beta Relationship")
    x_vals = np.linspace(index_returns.min(), index_returns.max(), 100)
    y_vals = alpha + beta * x_vals

    plt.plot(x_vals, y_vals, color=color, lw=2, label=f'y = {alpha:.4f} + {beta:.2f}x')
    cyberpunk.make_lines_glow(ax)

    plt.axhline(0, color="white", linestyle="--", alpha=0.2 )
    plt.axvline(0, color="white", linestyle="--", alpha=0.2 )
    
    # remove background
    if discord:
        fig.patch.set_alpha(0.0)       
        ax.set_facecolor("none")
    
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, dpi=300)
        buf.seek(0)
        plt.close(fig)
        return buf  # return the BytesIO buffer for Discord sending
    else:
        plt.legend()
        plt.show()
        return fig
   
