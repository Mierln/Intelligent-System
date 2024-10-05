# Importing libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

# Importing parameters
from parameters import *


def create_model(
    sequence_length,
    n_features,
    units=256,
    cell=LSTM,
    n_layers=2,
    dropout=0.3,
    loss="mean_absolute_error",
    optimizer="rmsprop",
    bidirectional=False,
):
    '''
    This function will create the model based on the parameters given.
    With the first layer and the last layer being slightly different from
    the hidden layers. The first layer will contain the input shape, while
    the last layer return_sequences will be set to False
    '''

    # Creating the model
    model = Sequential()

    # Looping for all of the numbers of layers
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(
                    Bidirectional(
                        cell(units, return_sequences=True),
                        batch_input_shape=(None, sequence_length, n_features),
                    )
                )
            else:
                model.add(
                    cell(
                        units,
                        return_sequences=True,
                        batch_input_shape=(None, sequence_length, n_features),
                    )
                )
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))

        # add dropout after each layer
        model.add(Dropout(dropout))

    # add Dense layer
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)

    return model


def predict(model, data, k_days):
    '''
    This function takes the trained model and data as the input
    and by using the parameters it will get the last sequence and
    predict that last sequence. If the data was scaled to begin with,
    it will inverse transform the data back to the original price.
    ''' 
    feature_columns = FEATURE_COLUMNS
    
    index_of_close = feature_columns.index("Close")
    index_of_open = feature_columns.index("Open")
    index_of_high = feature_columns.index("High")
    index_of_low = feature_columns.index("Low")
    index_of_volume = feature_columns.index("Volume")
    
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    
    predictions = []
    
    for i in range(k_days):
        # Make the prediction (scaled)
        prediction = model.predict(last_sequence)
        
        # Inverse transform the prediction to get the actual price
        if SCALE:
            predicted_price = data["column_scaler"]["Close"].inverse_transform(prediction)[0][0]
        else:
            predicted_price = prediction[0][0]
        # Append the predicted price to the predictions list
        predictions.append(predicted_price)
        
        # Scale the predicted price before adding it to last_sequence
        if SCALE:
            scaled_predicted_price = data["column_scaler"]["Close"].transform([[predicted_price]])[0][0]
        else:
            scaled_predicted_price = predicted_price
        
        last_known_values = last_sequence[-1]
    
        new_entry = np.copy(last_known_values)
        
        # Update the "Close" price
        new_entry[index_of_close] = scaled_predicted_price
        new_entry[index_of_open] = new_entry[index_of_close]
        
        percentage = 0.01  # 1% for example
        
        new_entry[index_of_high] = new_entry[index_of_close] * (1 + percentage)
        new_entry[index_of_low] = new_entry[index_of_close] * (1 - percentage)
        
        new_entry[index_of_volume] = last_known_values[index_of_volume]

        # Update last_sequence
        last_sequence = np.append(last_sequence[1:], [new_entry], axis=0)  

    # Print the future prices
    for idx, future_price in enumerate(predictions, 1):
        print(f"Future price after {idx} days is {future_price:.2f}$")
    
