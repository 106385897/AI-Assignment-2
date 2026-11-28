# LSTM model for traffic flow prediction (Keras).
# Run: python lstm_model.py

import json
import os
import time

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  
import tensorflow as tf                             
from tensorflow import keras                         

from data_processing import LOOKBACK, N_FEATURES, prepare_dataset  # noqa: E402

# ---- settings ----
UNITS = 64
DROPOUT = 0.2
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 42
MIN_FLOW_FOR_MAPE = 10       # ignore near-zero flows, where percentage error explodes
MODEL_PATH = "outputs/lstm_model.keras"
RESULTS_PATH = "outputs/lstm_results.json"
PREDICTIONS_PATH = "outputs/lstm_predictions.csv"


def build_lstm(lookback=LOOKBACK, n_features=N_FEATURES, units=UNITS, dropout=DROPOUT):
    # two LSTM layers, then a small dense layer that outputs one number
    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, n_features)),
        keras.layers.LSTM(units, return_sequences=True),
        keras.layers.Dropout(dropout),                  # dropout helps avoid overfitting
        keras.layers.LSTM(units // 2),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ], name="lstm")
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse",
                  metrics=["mae"])
    return model


def train_lstm(data, epochs=EPOCHS, batch_size=BATCH_SIZE):
    keras.utils.set_random_seed(SEED)                   # same results each run
    model = build_lstm(data.lookback)
    # stop when validation loss stops improving and keep the best weights
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True)
    history = model.fit(data.X_train, data.y_train,
                        validation_data=(data.X_val, data.y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[stop], verbose=2)
    return model, history


def evaluate(y_true, y_pred):
    # error scores in vehicles per 15 minutes
    err = y_true - y_pred
    mask = y_true >= MIN_FLOW_FOR_MAPE                  # only use flows of 10 or more for MAPE
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err[mask] / y_true[mask])) * 100),
        "R2": float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
    }


def predict_flow(model, data, X, sites):
    # predict, then convert back from 0-1 to vehicle counts
    scaled = model.predict(X, verbose=0).ravel()
    return data.inverse_transform(np.clip(scaled, 0, 1.5), sites)


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    data = prepare_dataset()
    start = time.time()
    model, history = train_lstm(data)
    train_seconds = time.time() - start                # how long training took

    # real values vs what the model predicted on the test days
    y_true = data.inverse_transform(data.y_test, data.site_test)
    y_pred = predict_flow(model, data, data.X_test, data.site_test)
    # simple baseline: guess the next value is the same as the last one
    y_naive = data.inverse_transform(data.X_test[:, -1, 0], data.site_test)

    results = {"LSTM": evaluate(y_true, y_pred),
               "Persistence baseline": evaluate(y_true, y_naive),
               "epochs_run": len(history.history["loss"]),
               "train_seconds": round(train_seconds, 1),
               "parameters": int(model.count_params())}
    print(json.dumps(results, indent=2))

    # save the test predictions (site, true, pred) for the comparison
    pd.DataFrame({"site": data.site_test, "true": y_true, "pred": y_pred}
                 ).to_csv(PREDICTIONS_PATH, index=False)

    # save the model and the scores
    model.save(MODEL_PATH)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)