import numpy as np
from data_loader import load_and_preprocess_data
from garch_model import fit_garch_model, add_garch_predictions
from utils import list_columns_to_dropna, get_lagged
from lstm import *
from features import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import platform

import ta
import pandas as pd
import os

# 根据操作系统设置中文字体
system = platform.system()
if system == 'Windows':
    font_list = ['SimHei', 'Microsoft YaHei', 'SimSun']
elif system == 'Darwin':  # MacOS
    font_list = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
else:  # Linux
    font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Noto Sans CJK JP']

# 尝试设置中文字体
for font in font_list:
    try:
        plt.rcParams['font.sans-serif'] = [font]
        # 测试字体
        plt.figure(figsize=(1,1))
        plt.text(0.5, 0.5, '测试')
        plt.close()
        print(f"成功使用字体: {font}")
        break
    except:
        continue

plt.rcParams['axes.unicode_minus'] = False

# 创建字体对象
try:
    font = FontProperties(family=plt.rcParams['font.sans-serif'][0])
except:
    # 如果上述字体都不可用，使用系统默认字体
    font = FontProperties()

def process(input_path, output_path, predict_path):
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

    df.to_csv(output_path)

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
    # plot_training_history(hist)

    # Evaluate model
    evaluate_model(lstm, X_test, y_test)
    print_predictions(lstm, X_test, y_test, predict_path)
    
    # Retrain model with full dataset
    print("\nStarting to retrain model with full dataset...")
    X_full = np.concatenate((X_train, X_test), axis=0)
    y_full = np.concatenate((y_train, y_test), axis=0)
    
    # Retrain model
    lstm_full = create_lstm_model((X_full.shape[1], X_full.shape[2]))
    hist_full = train_lstm_model(lstm_full, X_full, y_full)
    
    # Generate future predictions
    future_predictions = predict_future(lstm_full, X_full[-1:], 365)
    
    # Save future predictions
    save_future_predictions(future_predictions, output_path)

def predict_future(model, last_sequence, days_to_predict):
    """
    Predict future days_to_predict days of volatility based on the last sequence
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_to_predict):
        # Get next day's prediction
        next_pred = model.predict(current_sequence)[0][0]
        predictions.append(next_pred)
        
        # Update sequence for next prediction
        # Note: Here we assume the input features only include volatility, which should be adjusted according to your feature structure
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred
    
    return np.array(predictions)

def save_future_predictions(predictions, output_path):
    """
    Save future predictions to CSV file with input filename pattern
    """
    # Get the input filename without extension
    input_filename = os.path.splitext(os.path.basename(output_path))[0]
    
    start_date = datetime.now()
    dates = [start_date + pd.Timedelta(days=x) for x in range(len(predictions))]
    
    future_df = pd.DataFrame({
        'Date': dates,
        'Predicted_Volatility': predictions
    })
    
    # Create output paths with input filename pattern
    output_dir = os.path.dirname(output_path)
    future_predictions_path = os.path.join(output_dir, f'{input_filename}_future_predictions.csv')
    plot_path = os.path.join('Images', f'{input_filename}_future_predictions.png')
    
    # Save predictions to CSV
    future_df.to_csv(future_predictions_path, index=False)
    print(f"\nFuture 365-day predictions saved to: {future_predictions_path}")
    
    # Create and save plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, predictions, label='Predicted Volatility', color='blue', linewidth=2)
    plt.title(f'365-Day Volatility Forecast - {input_filename}', fontsize=12)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Volatility', fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(prop={'size': 10})
    plt.tight_layout()
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance and output detailed metrics
    """
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R-squared (R²): {r2:.6f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red')
    plt.title('Volatility Prediction Comparison')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.legend(prop={'size': 10})
    plt.grid(True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'Images/prediction_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Scatter Plot')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig(f'Images/scatter_plot_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_predictions(model, X_test, y_test, predict_path):
    """
    Save prediction results to CSV file
    """
    # Get the input filename without extension
    input_filename = os.path.splitext(os.path.basename(predict_path))[0]
    
    y_pred = model.predict(X_test)
    
    results_df = pd.DataFrame({
        'Actual_Value': y_test.flatten(),
        'Predicted_Value': y_pred.flatten(),
        'Prediction_Error': np.abs(y_test.flatten() - y_pred.flatten())
    })
    
    # Create output path with input filename pattern
    output_dir = os.path.dirname(predict_path)
    predictions_path = os.path.join(output_dir, f'{input_filename}_test_predictions.csv')
    
    results_df.to_csv(predictions_path, index=False)
    print(f"\nPrediction results saved to: {predictions_path}")

