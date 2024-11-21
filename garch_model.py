from arch import arch_model
import pandas as pd

def fit_garch_model(log_returns):
    GARCH_model = arch_model(log_returns, vol='Garch', p=1, q=1, rescale=False)
    model_fit = GARCH_model.fit(disp='off')
    return model_fit

def add_garch_predictions(df, model_fit, horizon):
    forecast_rolling = model_fit.forecast(horizon=len(df) - 50)
    GARCH_rolling_predictions = pd.DataFrame(
        forecast_rolling.variance.iloc[:, 0],
        index=df.index[50:],
        columns=['GARCH_rolling_predictions']
    )

    forecast_forward = model_fit.forecast(horizon=horizon)
    GARCH_forward_looking_predictions = pd.DataFrame(
        forecast_forward.variance,
        index=df.index[-horizon:],
        columns=['GARCH_forward_looking_predictions']
    )

    df = pd.concat([df, GARCH_rolling_predictions, GARCH_forward_looking_predictions], axis=1)
    df.fillna(0, inplace=True)
    return df
