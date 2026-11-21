# GRU model for traffic flow prediction (Keras).
# Author: Falaq Nadeem
# Run: python gru_model.py

# Uses the same data, settings and scoring as lstm_model.py,
# so the LSTM and GRU can be compared fairly.
import json
import os
import time

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   
import tensorflow as tf                               
from tensorflow import keras                          

from data_processing import LOOKBACK, N_FEATURES, prepare_dataset  
from lstm_model import evaluate, predict_flow                      

UNITS = 64
DROPOUT = 0.2
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 42
MODEL_PATH = "outputs/gru_model.keras"
RESULTS_PATH = "outputs/gru_results.json"
PREDICTIONS_PATH = "outputs/gru_predictions.csv"


def build_gru(lookback=LOOKBACK, n_features=N_FEATURES, units=UNITS, dropout=DROPOUT):
    
    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, n_features)),
        keras.layers.GRU(units, return_sequences=True),   
        keras.layers.Dropout(dropout),                    
        keras.layers.GRU(units // 2),                     
        keras.layers.Dropout(dropout),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),                            
    ], name="gru")
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse",
                  metrics=["mae"])
    return model


def train_gru(data, epochs=EPOCHS, batch_size=BATCH_SIZE):
    keras.utils.set_random_seed(SEED)                     
    model = build_gru(data.lookback)
    
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True)
    start = time.time()
    history = model.fit(data.X_train, data.y_train,
                        validation_data=(data.X_val, data.y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[stop], verbose=2)
    train_seconds = time.time() - start                   
    return model, history, train_seconds


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    data = prepare_dataset()
    model, history, train_seconds = train_gru(data)

    y_true = data.inverse_transform(data.y_test, data.site_test)
    start = time.time()
    y_pred = predict_flow(model, data, data.X_test, data.site_test)
    predict_seconds = time.time() - start
    
    y_naive = data.inverse_transform(data.X_test[:, -1, 0], data.site_test)

    results = {"GRU": evaluate(y_true, y_pred),
               "Persistence baseline": evaluate(y_true, y_naive),
               "epochs_run": len(history.history["loss"]),
               "train_seconds": round(train_seconds, 1),
               "predict_seconds": round(predict_seconds, 2),
               "parameters": int(model.count_params())}
    print(json.dumps(results, indent=2))

    model.save(MODEL_PATH)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame({"site": data.site_test,
                  "true": y_true,
                  "pred": y_pred,
                  "naive": y_naive}).to_csv(PREDICTIONS_PATH, index=False)
    print(f"Saved model to {MODEL_PATH}, scores to {RESULTS_PATH}, "
          f"predictions to {PREDICTIONS_PATH}")