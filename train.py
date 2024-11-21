import numpy as np
from data_loader import load_and_preprocess_data
from garch_model import fit_garch_model, add_garch_predictions
from utils import list_columns_to_dropna, get_lagged
from lstm import *
from features import *

import ta
import pandas as pd

def process(input_path, output_path):
    df = load_and_preprocess_data(input_path)

    # Calculate Technical Indicators

    # Calculate SMA14
    df['SMA14'] = df['Close'].rolling(window=14).mean()

    # Calculate EMA14
    df['EMA14'] = df['Close'].ewm(span=14, adjust=False).mean()

    # Calculate RSI14
    df['RSI14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()

    # Calculate MACD and its signal line
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD_h'] = macd.macd_diff()  # MACD histogram
    df['MACD_sl'] = macd.macd_signal()  # MACD signal line

    # Fit GARCH model
    garch_model = fit_garch_model(df['Log_Returns'])
    df = add_garch_predictions(df, garch_model, horizon=1500)
    
    # Drop NaNs
    columns_to_check = [
        'Open', 'Log_Returns', 'Previous_10_Day_Volatility', 
        'Next_10_Days_Volatility', 'Previous_30_Day_Volatility',
        'SMA14', 'EMA14', 'RSI14', 'MACD_h', 'MACD_sl'
    ]
    df = list_columns_to_dropna(df, columns_to_check)

    df.to_csv('output.csv')

    # Prepare data
    features_to_drop = ['Next_10_Days_Volatility', 'Low', 'High', 'Close', 'Open', 'Volume', 'MACD_h','MACD_sl','RSI14','SMA14','EMA14']
    X = np.array(df.drop(features_to_drop, axis=1).values)
    y = np.array(df["Next_10_Days_Volatility"].values).reshape(-1, 1)
    test_size = 1500
    X_train, X_test = X[test_size:], X[:test_size]
    y_train, y_test = y[test_size:], y[:test_size]

    N = 30
    X_train, y_train = get_lagged(X_train, y_train, N, (X_train.shape[0]-N, N*X_train.shape[1]))
    X_test, y_test = get_lagged(X_test, y_test, N, (X_test.shape[0]-N, N*X_test.shape[1]))
    T = 4
    X_train, y_train = get_lagged(X_train, y_train, T, (X_train.shape[0]-T, T, X_train.shape[1]))
    X_test, y_test = get_lagged(X_test, y_test, T, (X_test.shape[0]-T, T, X_test.shape[1]))

    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    # Train LSTM model
    lstm = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    hist = train_lstm_model(lstm, X_train, y_train)
    plot_training_history(hist)

    # Evaluate model
    evaluate_model(lstm, X_test, y_test)
    print_predictions(lstm, X_test, y_test)

