import yfinance as yf
import pandas as pd
from os.path import exists

from sklearn.model_selection import train_test_split


def load_data():
    
    company = input("Enter Company Name: ")
    start = input("Enter Start Date (YYYY-MM-DD): ")
    end = input("Enter Start Date (YYYY-MM-DD): ")
    
    path = f"./saved_data/{company}_{start}_{end}.csv"
    
    if exists(path):
        data = pd.read_csv(path)
    
    else:
        data = yf.download(company, start, end)
        save = input("Do you want to save? (y/n): ")
        
        if save.upper() == "Y":
            data.to_csv(path)
            
    return data


def fill_nan(df):
    
    for col in df.columns:
        # Iterate through the DataFrame column by column
        for i in range(1, len(df) - 1):
            if pd.isna(df.at[i, col]):
                # Calculate the average of the previous and next non-NaN values
                before = df.at[i - 1, col]
                after = df.at[i + 1, col]
                if not pd.isna(before) and not pd.isna(after):
                    df.at[i, col] = (before + after) / 2
                    
                    
    return df


def split_mode(data):
    # split by ratio train test split
    
    
    # split by date
    
    # split by random
    
    pass

def choose_scaler():
    pass


print(load_dataset())