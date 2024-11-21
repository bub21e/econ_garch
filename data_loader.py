import pandas as pd
import numpy as np

def load_and_preprocess_data(input_path):
    df = pd.read_csv(input_path)
    df = df[['日期_Date', '开盘价_Oppr', '收盘价_Clpr', '最高价_Hipr', '最低价_Lopr', '成交量_Trdvol']]
    df = df.rename(columns={
        '日期_Date': 'Date',
        '开盘价_Oppr': 'Open',
        '收盘价_Clpr': 'Close',
        '最高价_Hipr': 'High',
        '最低价_Lopr': 'Low',
        '成交量_Trdvol': 'Volume'
    })
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)

    # Feature engineering
    df['Log_Returns'] = np.log(df.Close) - np.log(df.Close.shift(1))
    df['Log_Trading_Range'] = np.log(df.High) - np.log(df.Low)
    df['Log_Volume_Change'] = np.log(df.Volume) - np.log(df.Volume.shift(1))
    df['Previous_10_Day_Volatility'] = df['Log_Returns'].rolling(window=10).std()
    df['Previous_30_Day_Volatility'] = df['Log_Returns'].rolling(window=30).std()
    df['Next_10_Days_Volatility'] = df['Log_Returns'].iloc[::-1].rolling(window=10).std().iloc[::-1]
    df.dropna(inplace=True)
    return df
