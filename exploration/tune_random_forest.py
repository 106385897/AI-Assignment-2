import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from models.random_forest import RandomForestTraffic
from train_random_forest import load_data, prepare_data


def main():

    print("Loading traffic data...")

    traffic = load_data()

    # Reserve the final testing period
    development_data = traffic[
        traffic["Datetime"] < pd.Timestamp("2006-10-25")
    ].copy()

    # Use 1-20 October for training and 21-24 for validation
    X_train, X_val, y_train, y_val = prepare_validation_data(
        development_data
    )

    configurations = [
        (50, 10, 5),
        (100, 15, 5),
        (100, 20, 5),
        (150, 20, 2)
    ]

    results = []

    for trees, depth, samples in configurations:

        print("\nTesting configuration:")
        print(f"Trees: {trees}")
        print(f"Maximum depth: {depth}")
        print(f"Minimum samples to split: {samples}")

        model = RandomForestTraffic(
            n_estimators=trees,
            max_depth=depth,
            min_samples_split=samples,
            random_state=42
        )

        model.train(X_train, y_train)

        predictions = model.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)
        rmse = np.sqrt(
            mean_squared_error(y_val, predictions)
        )
        r2 = r2_score(y_val, predictions)

        results.append({
            "Trees": trees,
            "Max Depth": depth,
            "Min Samples Split": samples,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAE")

    print("\nHyperparameter Comparison")
    print("-------------------------")
    print(results_df.to_string(index=False))

    print("\nBest Configuration:")
    print(results_df.iloc[0].to_string())

    results_df.to_csv(
        "random_forest_tuning_results.csv",
        index=False
    )


def prepare_validation_data(traffic):

    traffic["Hour"] = traffic["Datetime"].dt.hour
    traffic["Minute"] = traffic["Datetime"].dt.minute
    traffic["DayOfWeek"] = traffic["Datetime"].dt.day_of_week

    grouped = traffic.groupby("SCATS Number")

    for i in range(1, 5):

        traffic[f"Lag_{i}"] = grouped["Volume"].shift(i)

        previous_time = grouped["Datetime"].shift(i)

        valid_interval = (
            traffic["Datetime"] - previous_time
            == pd.Timedelta(minutes=15 * i)
        )

        traffic[f"Lag_{i}"] = traffic[f"Lag_{i}"].where(
            valid_interval
        )

    traffic = traffic.dropna()

    features = [
        "SCATS Number",
        "Hour",
        "Minute",
        "DayOfWeek",
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_4"
    ]

    split_date = pd.Timestamp("2006-10-21")

    train_data = traffic[
        traffic["Datetime"] < split_date
    ]

    validation_data = traffic[
        traffic["Datetime"] >= split_date
    ]

    X_train = train_data[features].to_numpy()
    y_train = train_data["Volume"].to_numpy()

    X_val = validation_data[features].to_numpy()
    y_val = validation_data["Volume"].to_numpy()

    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    main()