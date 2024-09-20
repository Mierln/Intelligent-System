# Importing all the files
from data_processing import load_data, get_final_df
from model import create_model, predict
from plotting import plotit, plot_graph
from parameters import *


def main():
    """
    This function is the main function that when run will go over data retrieving
    data processing, model creation, prediction and plotting of results.
    In the end this function also evaluate the model using an accuracy score
    as well as calculating the hypothetical profit.
    """

    # Loading the data
    data = load_data(
        N_STEPS,
        scale=SCALE,
        split_by_date=SPLIT_BY_DATE,
        shuffle=SHUFFLE,
        lookup_step=LOOKUP_STEP,
        test_size=TEST_SIZE,
        feature_columns=FEATURE_COLUMNS,
    )

    # Plotting the data in candlestick format
    # plotit(data["path"], n_days=30)

    # Creating the model with the parameters imported from the parameters file
    model = create_model(
        N_STEPS,
        len(FEATURE_COLUMNS),
        loss=LOSS,
        units=UNITS,
        cell=CELL,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        optimizer=OPTIMIZER,
        bidirectional=BIDIRECTIONAL,
    )

    # Training the model with parameters imported from the parameters file
    model.fit(
        data["X_train"],
        data["y_train"],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(data["X_test"], data["y_test"]),
        verbose=1,
    )

    # evaluate the model
    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)

    # calculate the mean absolute error (inverse scaling)
    if SCALE:
        mean_absolute_error = data["column_scaler"]["Close"].inverse_transform([[mae]])[
            0
        ][0]
    else:
        mean_absolute_error = mae

    # get the final dataframe for the testing set
    final_df = get_final_df(model, data)
    
    

    # we calculate the accuracy by counting the number of positive profits
    accuracy_score = (
        len(final_df[final_df["sell_profit"] > 0])
        + len(final_df[final_df["buy_profit"] > 0])
    ) / len(final_df)
    
    # calculating total buy & sell profit
    total_buy_profit = final_df["buy_profit"].sum()
    total_sell_profit = final_df["sell_profit"].sum()
    
    # total profit by adding sell & buy together
    total_profit = total_buy_profit + total_sell_profit
    
    # dividing total profit by number of testing samples (number of trades)
    profit_per_trade = total_profit / len(final_df)
    
    # printing metrics
    print("")
    
    predict(model, data, k_days=5)
    
    print("")
    print(f"{LOSS} loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    print("Accuracy score:", accuracy_score)
    print("Total buy profit:", total_buy_profit)
    print("Total sell profit:", total_sell_profit)
    print("Total profit:", total_profit)
    print("Profit per trade:", profit_per_trade)

    # plot true/pred prices graph
    plot_graph(final_df)


if __name__ == "__main__":
    main()
