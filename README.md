# LSTM Stock Price Prediction

This project uses an LSTM (Long Short-Term Memory) neural network to predict the next day's closing price of a stock based on historical data.

The example in this project uses Tesla (TSLA) stock data from 2015 to 2024, fetched using the Yahoo Finance API.

## Features

- Fetches historical stock price data
- Scales the data using MinMaxScaler
- Trains an LSTM model on the last 60 days of stock prices
- Predicts the next day's closing price
- Plots the historical closing price chart

## Technologies Used

- Python
- NumPy
- Pandas
- yfinance
- Matplotlib
- scikit-learn
- TensorFlow / Keras
