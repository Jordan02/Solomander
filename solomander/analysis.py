import pandas as pd


def max_drawdown(pnl: pd.Series):
    
    pnl_list = pnl.to_numpy()
    max_loss = 0
    current_loss = 0

    for val in pnl_list:
        if val < 0:
            current_loss += val  # add loss
            max_loss = min(max_loss, current_loss)  # track the most negative
        else:
            current_loss = 0  # reset on win
    return max_loss
