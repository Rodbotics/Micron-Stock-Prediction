# START OF DOCUMENTATION SECTION #

# PROGRAM NAME: Micron Stock Prediction - LSTM Regression Model

# SUMMARY OF PROGRAM:
# This program creates and trains a LSTM (RNN) model to predict the future stock price of Micron. Specifically,...
# this program takes a years worth of Micron's stock data -- 252 trading days -- and projects the closing stock price for the
# next 60 days (2 months). As a result, training an LSTM model can indicate whether the stock price will trend positively or negatively.

# Since I am forecasting Micron's stock price, LSTM is implemented as regression

# PARAMETERS:
# For each LSTM cell, the model must train 8 parameters: 4 weight matrices and 4 bias vectors

# LSTM STEPS:
# In input gate, there is a FC layer with sigmoid and tanh activation functions #
# In forget gate, there is a FC layer with sigmoid activation function
# An updated cell state or C state is determined based on combining the input and forget gate information
# That is, the input gate determines if cell is updated and the forget gate determines how much of the old state should be forgotten or removed
# Finally, the hidden (h) state or vector is found by multiplying the output gate with the tanch of the updated cell state #
# Recurrent cells are called as such because the hidden state and cell state of the previous LSTM cell influence the next cells' C & H states

# Suffice it to say, the cell remembers the most pertinent stock prices over set number of days
# and the three gates regulate the flow of information into and out of cells

# WHY IS LSTM VIABLE FOR STOCK PRICE PREDICTION?
# LSTM is applicable for stock price prediction is because stock prices are a form of time series data on a discrete... #
# time scale (price changes per day) . That is to say I can observe stock prices change sequentially in time. #
# LSTM is also good because it has the ability to store previous (pertinent) stock price to predict its future stock price #

# END OF DOCUMENTATION #

############################################################################################################################################

# START OF PROGRAM #

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
plt.style.use('fivethirtyeight')

##################################
# READING FILE - IMPORTING DATASET

df = pd.read_csv('Micron_LSTM.csv')                           # Micron stock quote from Yahoo finance
# print(df.head(), '\n')                                      # Displays first 5 rows of stock data
# print(df.tail(), '\n')                                      # Displays last 5 rows of stock data
# print(df.shape, '\n')                                       # Returns shape of stock data --> [252x7]

###########################################
# PRE-PROCESSING DATA PART 1: NORMALIZATION

# I only use the 'Close' feature because I want to predict the closing price of the stock. Not the opening, low, high, etc.

data = df.filter(['Close'])                                 # Create a new dataframe with only 'Close' column
dataset = data.values                                       # Convert 'Close' dataframe to a numpy array
# print(dataset, '\n')                                        # [252x1] original closing price data

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))                 # Created normalization object
scaled_data = scaler.fit_transform(dataset)                 # Apply normalization to our dataset --> range = [0,1] inclusive
print('The dimensions for our scaled data is:', scaled_data.shape[0], 'by', scaled_data.shape[1], '\n')

##########################################################################
# PRE-PROCESSING DATA PART 2: SPLIT DATASET INTO TRAINING AND TESTING SETS

# Create the rows of training dataset for the LSTM model
training_ds_rows = math.ceil(len(scaled_data) * 0.70)              # Try 65%, 70%, and 80% training set size
print('The number of rows in training set =', training_ds_rows, '\n')

# Create the rows of test dataset for the LSTM model
testing_ds_rows = len(scaled_data) - training_ds_rows
print('The number of rows in testing set =', testing_ds_rows, '\n')

# Create the scaled training set for the LSTM model
scaled_training_set = scaled_data[0:training_ds_rows, :]
# print(scaled_training_set, '\n')

# Create the scaled test set for the LSTM model
scaled_test_set = scaled_data[training_ds_rows:len(scaled_data), :]
# print(scaled_test_set.shape, '\n')


# Function for X_train, y_train, X_test, y_train
def create_dataset(ds, time_step):
    # Initializing X_data and y_data
    X_data = []
    y_data = []
    # Looping for entire length of scaled_training_set or scaled_test_set
    for i in range(len(ds) - time_step-1):
        # Shifting X_data and y_data values by +1 for every iteration 'i'
        X_data.append(ds[i:(i + time_step), 0])
        y_data.append(ds[i + time_step, 0])
    # Convert X_train and y_train to numpy arrays
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data


# Time_step is synonymous with number of days to look_back and n_steps
time_step = 10

# Creating X_train and y_train from scaled training set
X_train, y_train = create_dataset(scaled_training_set, time_step)
# print(X_train.shape, '\n')                                        # Size: [1 x time_step]
# print(y_train.shape, '\n')                                        # Last number of X_train becomes 2nd to last in y_train. Size: [Nx1]

# Creating X_test and y_test from scaled test set
X_test, y_test = create_dataset(scaled_test_set, time_step)
# print(X_test.shape, '\n')                                         # Size: [N time_step]
# print(y_test.shape, '\n')                                         # Last number of X_test becomes 2nd to last in y_test. Size: [Nx1]

# Reshape input data for LSTM model by adding an extra dimension
# Data should be reshaped to: (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#############################
# CREATE "STACKED" LSTM MODEL

# Model architecture: 2 LSTM layers + 1 Dense (FC) layers
# Note: By default LSTM layer includes the necessary activation (tanh) and recurrent activation (sigmoid) functions
model = Sequential(
    [
        LSTM(6, input_shape=(X_train.shape[1], 1), return_sequences=True),
        LSTM(6, return_sequences=False),
        Dense(1)
     ]
)

# 'None' appears in output shape because batch size is not known at this point
print(model.summary(), '\n')

# Goal: Choose the best optimizer to reduce loss function. Small loss function = model was trained well
# Loss function = MSE & optimizer = 'adam' (variant of SGD)
LSTM_compiled = model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model

# Training LSTM model
# Pass validation data to evaluate validation loss and other metrics at the end of each epoch
LSTM_model = model.fit(X_train, y_train, batch_size=12, epochs=35, verbose=1, validation_data=(X_test, y_test))
print(LSTM_model, '\n')

####################################################
# PREDICT TEST DATA, EVALUATE MODEL, AND PLOT OUTPUT

# Prediction and performance metric values
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Undo scaling of data to match with y_test and y_train values
predict_train = scaler.inverse_transform(predict_train)
predict_test = scaler.inverse_transform(predict_test)

# Evaluating model by calculating RMSE. Generally, smaller RMSE = better fit
RMSE_train = math.sqrt(mean_squared_error(y_true=y_train, y_pred=predict_train))
RMSE_test = math.sqrt(mean_squared_error(y_true=y_test, y_pred=predict_test))
print('The RMSE error for training set = ' + str(RMSE_train))
print('The RMSE error for testing set = ' + str(RMSE_test) + '\n')

# Calculating the mean forecast error
forecast_error = predict_test - y_test
MFE = np.mean(forecast_error)
print('The mean forecast error = ' + str(MFE), '\n')

# Plotting
look_back = time_step                                       # Number of days to observe closing stock price, from the last datapoint
# Shifting train predictions
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(predict_train)+look_back, :] = predict_train
# print(train_predict_plot, '\n')                             # Predicted closing prices from trained data

# Shifting test predictions - Predicted closing prices
test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[look_back:len(predict_test)+look_back, :] = predict_test
# print(test_predict_plot, '\n')                               # Predicted closing prices from test data

# Plotting baseline and predictions
plt.figure(figsize=(16, 8))
plt.title('Micron Stock Closing Price')
plt.xlabel('Date (# of Days)', fontsize=18)
plt.ylabel('Closing Price ($)', fontsize=18)
plt.plot(scaler.inverse_transform(scaled_data))
plt.plot(train_predict_plot)                                # Predicted closing prices from trained data
plt.plot(test_predict_plot)                                 # Predicted closing prices from test data
plt.legend(['Original', 'Predicted trained', 'Predicted Test'], loc='lower right')

################################################
# PREDICT THE FUTURE 60 DAYS AND PLOT THE OUTPUT

x_input = scaled_test_set[len(scaled_test_set)-time_step:].reshape(1, -1)       # Previous 'N' time step test, size: [1 x time_steps]
# print(x_input.shape, '\n')                                  # Size: [1 x time_steps]

temp_input = list(x_input)                                  # Converting array to list
# print(temp_input, '\n')
temp_input = temp_input[0].tolist()                         # Converting index to list

n_steps = time_step
days2forecast = 60                                          # Number of days I want prediction of stock closing price
final_output = []                                           # Future closing price output
i = 0                                                       # Loop counter

# Forecasting closing stock price 60 days into the future
while i < days2forecast:
    # Passing past 'N' days (time step) into our model for prediction
    if len(temp_input) > n_steps:
        # Shifting closing price to the right (+1)
        x_input = np.array(temp_input[1:])
        # print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        y_hat = model.predict(x_input, verbose=0)
        # print("{} day output {}".format(i, y_hat))
        temp_input.extend(y_hat[0].tolist())
        temp_input = temp_input[1:]
        final_output.extend(y_hat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        y_hat = model.predict(x_input, verbose=0)
        # print(y_hat[0])                                   # First predicted closing price value
        temp_input.extend(y_hat[0].tolist())
        # print(len(temp_input))                            # Number of closing price predictions
        final_output.extend(y_hat.tolist())
        i = i + 1

# print(final_output, '\n')                                   # Normalized closing prices

day_new = np.arange(1, n_steps+1)
day_pred = np.arange(n_steps+1, n_steps+1+days2forecast)

# Comparing Micron's actual closing price to the 60 day predicted price
plt.figure(figsize=(16, 8))
plt.title('Micron Actual vs. Predicted Closing Price')
plt.xlabel('Date (# of Days)', fontsize=18)
plt.ylabel('Closing Price ($)', fontsize=18)

final_output = scaler.inverse_transform(final_output)
for i in range(0, days2forecast):                           # Displays 60 days of predicted Micron stock closing price
    print("Micron's predicted closing stock price for day", i+1, "is:", final_output[i])

print('\n--------------------------------------------------------------------\n')

actual_output = scaler.inverse_transform(scaled_data[len(scaled_data)-n_steps:])
for i in range(0, n_steps):                                 # Displays past 'N' time step days of Actual Micron stock closing price
    print("Micron's actual closing stock price for the past", n_steps,"days... For day", i+1, "the price is:", actual_output[i])

plt.plot(day_new, actual_output)
plt.plot(day_pred, final_output)
plt.legend(['Actual Closing Price', 'Predicted Closing Price'], loc='upper left')

# Combining actual and predicted closing price lines together on one plot
plt.figure(figsize=(16, 8))
plt.title('Combined Actual and Predicted Closing Prices for Micron')
plt.xlabel('Date (# of Days)', fontsize=18)
plt.ylabel('Closing Price ($)', fontsize=18)
scaled_data2 = scaled_data.tolist()
final_output = scaler.fit_transform(final_output)
scaled_data2.extend(final_output)
plt.plot(scaled_data2[193:])
plt.show()

# END OF PROGRAM #

############################################################################################################################################
