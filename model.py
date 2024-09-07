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


def predict(model, data):
    '''
    This function takes the trained model and data as the input
    and by using the parameters it will get the last sequence and
    predict that last sequence. If the data was scaled to begin with,
    it will inverse transform the data back to the original price.
    ''' 
    
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)

    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["Close"].inverse_transform(prediction)[
            0
        ][0]
    else:
        predicted_price = prediction[0][0]

    return predicted_price
