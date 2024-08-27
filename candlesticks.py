# Importing Libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mplfinance as fplt


def resample_stock_data(df, n_days):
    # Compute the group index
    group_index = np.arange(len(df)) // n_days
    
    # Resampling logic
    resampled_data = {
        'Date': df.groupby(group_index)['Date'].first(),
        'Open': df.groupby(group_index)['Open'].first(),
        'Close': df.groupby(group_index)['Close'].last(),
        'High': df.groupby(group_index)['High'].max(),
        'Low': df.groupby(group_index)['Low'].min()
    }

    return pd.DataFrame(resampled_data).reset_index(drop=True)

def plot_candle(path, n_days=1):
    
    
    title = path.split("/")[-1]
    title = title.split("_")
    title = f"Price of {title[0]} from {title[1]} to {title[2][:-4]}"
    
    df = pd.read_csv(path)
    
    if  n_days >= 1:
        df = resample_stock_data(df, n_days=n_days)

    df.index = pd.DatetimeIndex(df['Date'])

    marketcolors = fplt.make_marketcolors(
        up='g',
        down='r',
        edge={'up': 'g', 'down': 'r'},
        wick={'up': 'g', 'down': 'r'},
        ohlc='black',
    )
    
    style  = fplt.make_mpf_style(marketcolors=marketcolors, gridstyle = 'solid', gridcolor = 'f0f0f0')

    fplt.plot(
            df,
            type='candle',
            title=title,
            ylabel='Price ($)',
            style = style,
            )

# plot_candle("./saved_data/CBA.AX_2024-01-01_2024-08-01.csv")