import yfinance as yf
import pandas as pd
import numpy as np
from os.path import exists
from collections import deque

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data(n_steps=60, scale=True, shuffle=True, lookup_step=1, split_by_date=True, test_size=0.2, feature_columns=["Adj Close", "Close", "Open", "High", "Low"]):
    """
    Load data from a time range that is given by input,
    if the data has already exists it will load that data
    if the data has not already exist it will fetch the yf and
    save it to a csv file if it's specified.
    """

    result = {}  # what will be returned
    data = get_data() # Get data from user input

    result["df"] = data.copy() # Copying raw data to be returned as well

    # Add date as a column
    if "date" not in data.columns:
        data["date"] = data.index

    # Applying scaler function
    if scale:
        column_scaler = {}
        for column in feature_columns:
            column_scaler[column], data[column] = scaler_function(data[column])

        # Adding to result
        result["column_scaler"] = column_scaler

    # Add the target column by shifting (default value 1)
    data["future"] = data["Close"].shift(-lookup_step)

    last_sequence = np.array(data[feature_columns].tail(lookup_step))
    
    data.dropna(inplace=True)

    sequence_data = []  # Creating sequence data
    sequences = deque(maxlen=n_steps)  # This sequence will be used as input to the LSTM model.

    for entry, target in zip(data[feature_columns + ["date"]].values, data["future"].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result['last_sequence'] = last_sequence

    # Creating sequence and target for training the dataset
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)


    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:    
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)
        
    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result

def get_data():
    '''
    Getting data by user input, if data is already saved then load that data
    if not prompt user if data should be saved or not
    '''
    
    company = input("Enter Company Name: ")
    start = input("Enter Start Date (YYYY-MM-DD): ")
    end = input("Enter Start Date (YYYY-MM-DD): ")
    
    path =  f"./saved_data/{company}_{start}_{end}.csv"
    
    if exists(path):
        data = pd.read_csv(path)

    else:
        data = yf.download(company, start, end)
        save = input("Do you want to save? (Y/n): ")

        if save.upper() == "Y" or save.upper() == "":
            data.to_csv(path)

    return data

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def scaler_function(column):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.expand_dims(column.values, axis=1))
    
    # Returning the scaler to inverse fit
    # Returning the scaled dataframe
    return scaler, scaled_data


print(load_data())