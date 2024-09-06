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

def plotit(path, n_days=1, ptype='candlestick'):
    title = path.split("/")[-1]
    title = title.split("_")
    title = f"Price of {title[0]} from {title[1]} to {title[2][:-4]}"
    
    df = pd.read_csv(path)
    
    if  n_days >= 1:
        df = resample_stock_data(df, n_days=n_days)
        
    df.index = pd.DatetimeIndex(df['Date'])
    
    if ptype == 'candlestick':
        plot_candle(df, title)
    elif ptype == 'boxplot':
        plot_boxplot(df, title)


def plot_candle(resampled_data, title):

    # https://github.com/matplotlib/mplfinance/blob/master/examples/styles.ipynb
    
    # Setting up the market colors (the candlestick)
    marketcolors = fplt.make_marketcolors(
        up='g',
        down='r',
        edge={'up': 'g', 'down': 'r'},
        wick={'up': 'g', 'down': 'r'},
        ohlc='black',
    )
    
    # Setting the background color
    style  = fplt.make_mpf_style(marketcolors=marketcolors, gridstyle = 'solid', gridcolor = 'f0f0f0')

    # plotting the candlestick
    fplt.plot(
            resampled_data,
            type='candle',
            title=title,
            ylabel='Price ($)',
            style = style,
            )
    
    
def plot_boxplot(resampled_data, title):
    boxplot_data = []
    
    # Gather data for each interval
    for i in range(len(resampled_data)):
        data_group = pd.concat([resampled_data.iloc[i][['Open', 'Close', 'High', 'Low']]])
        boxplot_data.append(data_group.values.flatten())  # Flatten to make sure it's a 1D array
    
    # Plot the boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(boxplot_data, patch_artist=True)
    plt.title(title)
    plt.xlabel('Interval')
    plt.ylabel('Price')
    plt.show()

# Example usage
plotit("./saved_data/CBA.AX_2024-01-01_2024-08-01.csv", n_days=1)

