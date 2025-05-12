import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Download stock data
stock = 'TSLA'  # You can change this to any stock
df = yf.download(stock, start='2015-01-01', end='2024-01-01')
data = df['Close'].values.reshape(-1, 1)

# Step 2: Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Step 3: Prepare training data
X, y = [], []
window_size = 60  # Use past 60 days to predict next day

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # [samples, time_steps, features]

# Step 4: Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=5, batch_size=32)

# Step 5: Make predictions on last 60 days
last_60_days = scaled_data[-60:]
X_test = np.array([last_60_days])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)

print(f"\n📊 Predicted Next Closing Price of {stock}: ${predicted_price[0][0]:.2f}")

# Step 6: Plot historical close prices
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Actual Closing Prices')
plt.title(f'{stock} Stock Closing Price History')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()
