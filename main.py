'''
NOTE: THIS IS NOT TO BE USED FOR STOCK PREDICTION BUT ONLY FOR MY MACHINE LEARNING TRAINING PURPOSES.
Using Recurring Neural Network (RNN) Approach
LSTM (Long Short Term Memory)
'''
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import ModelCheckpoint, TensorBoard
from data_preperation import loadData

gpus = tf.config.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
# Setting Seed
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

# Consts for training the model
N_STEPS = 50
LOOKUP_STEPS = 1 # Amount of days in the future you want to see the value of the stock chosen
SCALE = True
SHUFFLE = True
SPLIT_BY_DATE = False
TEST_SIZE = 0.2 # 20%
DROPOUT = 0.4 # 40%
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
N_LAYERS = 2
CELL = LSTM
BIDIRECTIONAL = False
UNITS = 256
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 500
TICKER = "NFLX" # Stock you want to predict


'''
Model Creation
'''

def createModel(sequence_length, n_features, units = 256, cell = LSTM, n_layers = 2, dropout = 0.3, loss = "mean_absolute_error", 
                optimizer = "rmsprop", bidirectional = False):
    model = Sequential()
    
    for i in range(n_layers):
        if i == 0:
            # First Layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # Final layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # Hidden layers
            if bidirectional:
                model.add(bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
                
        # Add dropout after each layer
        model.add(Dropout(dropout))
    
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    
    return model
    
    
'''
Training of the Model
'''

scale_str = f"sc-{int(SCALE)}"
shuffle_str = f"sh-{int(SHUFFLE)}"
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"

date_now = time.strftime("%Y-%m-%d")

ticker_data_filename = os.path.join(f"{TICKER}_{date_now}.csv")

model_name = f"{date_now}_{TICKER}-{shuffle_str}-{scale_str}-{split_by_date_str}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEPS}-layers-{N_LAYERS}-units-{UNITS}"
    
if BIDIRECTIONAL:
    model_name += "-b"
    
# Save model to results folder    
if not os.path.isdir("results"):
    os.mkdir("results")
    
    
# Load data
data = loadData(TICKER, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, shuffle=SHUFFLE, lookup_steps=LOOKUP_STEPS, test_size=TEST_SIZE, 
                feature_columns=FEATURE_COLUMNS)

# Save DataFrame
data["df"].to_csv(ticker_data_filename)

model = createModel(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS, dropout=DROPOUT, optimizer=OPTIMIZER,
                    bidirectional=BIDIRECTIONAL)

checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard()

history = model.fit(data["X_train"], data["y_train"], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard], verbose=1)


def plotGraph(test_df):
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEPS}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEPS}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual price", "Predicted Price"])
    plt.show()
    
def getFinalDf(model, data):
    buy_profit = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    
    X_test = data["X_test"]
    y_test = data["y_test"]
    y_pred = model.predict(X_test)
    
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
        
    test_df = data["test_df"]
    test_df[f"adjclose_{LOOKUP_STEPS}"] = y_pred
    test_df[f"true_adjclose_{LOOKUP_STEPS}"] = y_test
    test_df.sort_index(inplace=True)
    
    final_df = test_df
    final_df["buy_profit"] = list(map(buy_profit, final_df["adjclose"], final_df[f"adjclose_{LOOKUP_STEPS}"], final_df[f"true_adjclose_{LOOKUP_STEPS}"]))
    final_df["sell_profit"] = list(map(sell_profit, final_df["adjclose"], final_df[f"adjclose_{LOOKUP_STEPS}"], final_df[f"true_adjclose_{LOOKUP_STEPS}"]))
    
    return final_df

def predict(model, data):
    last_sequence = data["last_sequence"][-N_STEPS:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    
    prediction = model.predict(last_sequence)
    
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
        
    return predicted_price


# Load optimal model weights
model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

# Evaluate model
loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)

if SCALE:
    mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
else:
    mean_absolute_error = mae

# Get Final DataFrame    
final_df = getFinalDf(model, data)

# Predict future price
future_price = predict(model, data)

# Calculate accuracy through positive profits
accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)


print(f"Future price after {LOOKUP_STEPS} days is {future_price:.2f}$")
print(f"{LOSS} loss:", loss)
print("Mean Absolute Error:", mean_absolute_error)
print("Accuracy score:", accuracy_score)


# Show Graph
plotGraph(final_df)
    
csv_filename = os.path.join(model_name + ".csv")
final_df.to_csv(csv_filename)