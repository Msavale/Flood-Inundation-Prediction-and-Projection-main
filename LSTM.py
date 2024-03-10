import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Example data
# Assuming we have a sequence of time-series data with 10 time steps and 1 feature at each time step
input_shape = (10, 1)  # Number of time steps and number of features

# Build the model
model = Sequential([
    LSTM(64, input_shape=input_shape),  # LSTM layer with 64 units
    Dense(1)  # Output layer with 1 neuron for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display the model summary
model.summary()
