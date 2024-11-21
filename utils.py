import numpy as np

def list_columns_to_dropna(df, column_list):
    for column in column_list:
        df = df[df[column].notna()]
    return df

def get_lagged(x, y, t, s):
    lagged = []
    for i in range(x.shape[0] - t):
        for k in range(t):
            lagged.append(x[i+k])
    lagged = np.array(lagged).reshape(s)
    return lagged, y[:lagged.shape[0],]
