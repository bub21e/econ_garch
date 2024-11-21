import numpy as np
from data_loader import load_and_preprocess_data
from garch_model import fit_garch_model, add_garch_predictions
from utils import list_columns_to_dropna, get_lagged
from lstm import *
from features import *

def process(input_path):
    df = load_and_preprocess_data(input_path)
    
    # Fit GARCH model
    garch_model = fit_garch_model(df['Log_Returns'])
    df = add_garch_predictions(df, garch_model, horizon=1500)
    
    # Drop NaNs
    columns_to_check = ['Open', 'Log_Returns', 'Previous_10_Day_Volatility', 'Next_10_Days_Volatility', 'Previous_30_Day_Volatility']
    df = list_columns_to_dropna(df, columns_to_check)

    # build_pearson_correlation_matrix_of_dataframe(20,20,df,"Next_10_Days_Volatility",0.2)
    
    # Prepare data
    X = np.array(df.drop(["Next_10_Days_Volatility", 'Low', 'High', 'Close', 'Open', 'Volume'], axis=1).values)
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
