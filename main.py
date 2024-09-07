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
    
    # predict the future price
    future_price = predict(model, data)

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
    print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
    print(f"{LOSS} loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    print("Accuracy score:", accuracy_score)
    print("Total buy profit:", total_buy_profit)
    print("Total sell profit:", total_sell_profit)
    print("Total profit:", total_profit)
    print("Profit per trade:", profit_per_trade)

    # plot true/pred prices graph

    plot_graph(final_df)

    print(final_df.tail(10))


if __name__ == "__main__":
    main()


"""
# Load the test data
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'

test_data = yf.download('CBA.AX',TEST_START,TEST_END)

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values

model_inputs = model_inputs.reshape(-1, 1)

model_inputs = scaler.transform(model_inputs)

x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

COMPANY = 'CBA.AX'

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")
"""
