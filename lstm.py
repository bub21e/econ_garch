from keras.models import Model
from keras.layers import Input, LSTM, Dense
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, r2_score

def create_lstm_model(input_shape):
    inputLSTM = Input(shape=input_shape)
    y = LSTM(200, return_sequences=True)(inputLSTM)
    y = LSTM(200)(y)
    y = Dense(1)(y)
    lstm = Model(inputs=inputLSTM, outputs=y)
    return lstm

def train_lstm_model(model, X_train, y_train):
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['root_mean_squared_error']
    )
    hist = model.fit(X_train, y_train, batch_size=700, epochs=80, verbose=1, validation_split=0.3, shuffle=False)
    return hist

def plot_training_history(hist):
    plt.plot(hist.history['root_mean_squared_error'])
    plt.plot(hist.history['val_root_mean_squared_error'])
    plt.title('Model Train vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("R2 score:\n")
    print(('{:.2f}'.format((100 * (r2_score(y_test, y_pred))))) + " %\n")
    print("RMSE:\n")
    print(math.sqrt(mean_squared_error(y_test, y_pred)))
    print('\nMean Squared Error:\n')
    print(mean_squared_error(y_test, y_pred))

def print_predictions(model, X_test, y_test, file_path, scale=100):
    """
    Writes predictions and actual values to a file.

    Parameters:
        model: The trained model to use for predictions.
        X_test: Test data features.
        y_test: Test data labels.
        file_path: Path to the output file.
        scale: Scaling factor for the predictions and actual values (default is 100).
    """
    predictions = model.predict(X_test)
    with open(file_path, 'w') as f:
        for ind, pred in enumerate(predictions):
            actual = y_test[ind]
            f.write(f'Prediction: {round(scale * pred[0], 2):.2f},    Actual Value: {round(scale * actual[0], 2):.2f}\n')

