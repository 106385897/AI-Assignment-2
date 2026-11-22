# Trains the LSTM and GRU with several random seeds and compares them.
# Shows whether the difference between the models is real or just luck.
# Author: Falaq Nadeem

import os
import time

import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   
from tensorflow import keras                          

import gru_model                                      
import lstm_model                                     
from data_processing import prepare_dataset           
from lstm_model import evaluate, predict_flow         

SEEDS = [1, 2, 3, 42, 100]
RESULTS_PATH = "outputs/seed_results.csv"    # one row per model per seed
SUMMARY_PATH = "outputs/seed_summary.csv"    # average and spread per model


def run_once(name, module, train_fn, data, y_true, seed):
    # train one model with one seed and return its scores
    keras.backend.clear_session()             # start fresh, free memory from the last model
    module.SEED = seed                        # the train function reads SEED from its own file
    start = time.time()
    result = train_fn(data)                   # train_lstm returns 2 values, train_gru returns 3
    seconds = time.time() - start
    model, history = result[0], result[1]
    y_pred = predict_flow(model, data, data.X_test, data.site_test)
    return {"model": name,
            "seed": seed,
            **evaluate(y_true, y_pred),
            "epochs_run": len(history.history["loss"]),
            "train_seconds": round(seconds, 1),
            "parameters": int(model.count_params())}


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    data = prepare_dataset()                  # prepare the data once, reuse it for every run
    y_true = data.inverse_transform(data.y_test, data.site_test)

    models = [("LSTM", lstm_model, lstm_model.train_lstm),
              ("GRU", gru_model, gru_model.train_gru)]
    rows = []
    for seed in SEEDS:
        for name, module, train_fn in models:
            print(f"\n=== {name}, seed {seed} ===")
            rows.append(run_once(name, module, train_fn, data, y_true, seed))
            pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)   # save after every run

    df = pd.DataFrame(rows)
    # average and standard deviation (how much results vary) for each model
    cols = ["MAE", "RMSE", "MAPE", "R2", "epochs_run", "train_seconds", "parameters"]
    summary = df.groupby("model")[cols].agg(["mean", "std"]).round(3)
    summary.to_csv(SUMMARY_PATH)

    # for each seed, which model had the lower error?
    mae = df.pivot(index="seed", columns="model", values="MAE")
    gru_wins = int((mae["GRU"] < mae["LSTM"]).sum())

    print("\n===== Summary over", len(SEEDS), "seeds =====")
    print(summary)
    print(f"\nGRU had lower MAE than LSTM in {gru_wins} of {len(SEEDS)} seeds")