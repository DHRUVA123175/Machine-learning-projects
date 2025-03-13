import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

# Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2024-12-30")

# Feature Engineering
data["SMA_50"] = data["Close"].rolling(window=50).mean()
data["SMA_200"] = data["Close"].rolling(window=200).mean()
data["Prev_Close"] = data["Close"].shift(1)
data["Volume_Change"] = data["Volume"].pct_change()

# Drop NaN values after rolling window calculations
data.dropna(inplace=True)

# Define Features and Target
features = ["SMA_50", "SMA_200", "Prev_Close", "Volume_Change"]
X = data[features].values
y = data["Close"].values.reshape(-1, 1)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Scale Features and Target
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Reshape for LSTM
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Model Definition
model = Sequential([
    LSTM(units=250, activation="relu", return_sequences=True, input_shape=(X_train_reshaped.shape[1], 1)),
    Dropout(0.3),
    LSTM(units=125, activation="relu", return_sequences=True),
    Dropout(0.3),
    LSTM(units=75, activation="relu", return_sequences=False),
    Dropout(0.3),
    Dense(units=10, activation="relu"),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse")

# Train Model
model.fit(X_train_reshaped, y_train_scaled, epochs=300, batch_size=32, validation_data=(X_test_reshaped, y_test_scaled))

# Predictions
predictions_scaled = model.predict(X_test_reshaped)
predictions_original = scaler_y.inverse_transform(predictions_scaled)

# Convert Scaled y_test Back to Original Prices
y_test_original = scaler_y.inverse_transform(y_test_scaled)

# Plot Actual vs Predicted Prices
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test_original)), y_test_original, label="Actual Price", color="blue", linestyle="-")
plt.plot(range(len(predictions_original)), predictions_original, label="Predicted Price", color="orange", linestyle="--")

plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction")
plt.legend()
plt.show()
