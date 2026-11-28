import pandas as pd
import numpy as np

from models.random_forest import RandomForestTraffic
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


FILE_PATH = "data/Scats Data October 2006.xls"
SEQUENCE_LENGTH = 4


# Load and clean the traffic dataset
def load_data():

    df = pd.read_excel(
        FILE_PATH,
        sheet_name="Data",
        header=1
    )

    volume_columns = [f"V{i:02d}" for i in range(96)]

    df["SCATS Number"] = pd.to_numeric(
        df["SCATS Number"], errors="coerce"
    )

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    ).dt.normalize()

    df = df.dropna(subset=["SCATS Number", "Date"])
    df["SCATS Number"] = df["SCATS Number"].astype(int)

    for column in volume_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

        df[column] = df[column].where(df[column] >= 0)

    # Convert daily records into 15-minute observations
    traffic = df.melt(
        id_vars=["SCATS Number", "Date"],
        value_vars=volume_columns,
        var_name="Interval",
        value_name="Volume"
    )

    traffic["Interval"] = (
        traffic["Interval"].str[1:].astype(int)
    )

    traffic["Datetime"] = (
        traffic["Date"]
        + pd.to_timedelta(
            traffic["Interval"] * 15, unit="min"
        )
    )

    # Combine traffic measurements for each SCATS site
    traffic = traffic.groupby(
        ["SCATS Number", "Datetime"],
        as_index=False
    ).agg(
        Volume=("Volume", lambda x: x.sum(min_count=len(x)))
    )

    traffic = traffic.dropna(subset=["Volume"])

    traffic = traffic.sort_values(
        ["SCATS Number", "Datetime"]
    ).reset_index(drop=True)

    return traffic


# Create historical traffic features
def prepare_data(traffic):

    traffic["Hour"] = traffic["Datetime"].dt.hour
    traffic["Minute"] = traffic["Datetime"].dt.minute
    traffic["DayOfWeek"] = traffic["Datetime"].dt.day_of_week

    grouped = traffic.groupby("SCATS Number")

    for i in range(1, SEQUENCE_LENGTH + 1):

        traffic[f"Lag_{i}"] = grouped["Volume"].shift(i)

        previous_time = grouped["Datetime"].shift(i)

        valid_interval = (
            traffic["Datetime"] - previous_time
            == pd.Timedelta(minutes=15 * i)
        )

        traffic[f"Lag_{i}"] = traffic[f"Lag_{i}"].where(
            valid_interval
        )

    traffic = traffic.dropna().reset_index(drop=True)

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

    # Split chronologically to prevent future data leakage
    split_date = pd.Timestamp("2006-10-25")

    train_data = traffic[
        traffic["Datetime"] < split_date
    ]

    test_data = traffic[
        traffic["Datetime"] >= split_date
    ]

    X_train = train_data[features].to_numpy()
    y_train = train_data["Volume"].to_numpy()

    X_test = test_data[features].to_numpy()
    y_test = test_data["Volume"].to_numpy()

    return X_train, X_test, y_train, y_test


# Compare Random Forest against a persistence baseline
def compare_baseline(X_test, y_test, results):

    baseline_predictions = X_test[:, 4]

    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )

    baseline_r2 = r2_score(
        y_test, baseline_predictions
    )

    print("\nModel Performance Comparison")
    print("---------------------------------------------")
    print(f"{'Metric':<10}{'Random Forest':<18}{'Baseline'}")
    print(f"{'MAE':<10}{results['MAE']:<18.4f}{baseline_mae:.4f}")
    print(f"{'RMSE':<10}{results['RMSE']:<18.4f}{baseline_rmse:.4f}")
    print(f"{'R2':<10}{results['R2']:<18.4f}{baseline_r2:.4f}")

    if baseline_mae > 0:

        improvement = (
            (baseline_mae - results["MAE"])
            / baseline_mae
        ) * 100

        print(f"\nMAE improvement: {improvement:.2f}%")


def main():

    print("Loading SCATS traffic dataset...")

    traffic = load_data()

    print(f"Traffic observations: {len(traffic)}")
    print(f"SCATS sites: {traffic['SCATS Number'].nunique()}")

    X_train, X_test, y_train, y_test = prepare_data(traffic)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    model = RandomForestTraffic(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )

    model.train(X_train, y_train)

    results = model.evaluate(X_test, y_test)

    compare_baseline(X_test, y_test, results)

    model.save_model()

    predictions = model.predict(X_test)

    print("\nFirst 10 predictions:")
    print("Actual\tPredicted")

    for actual, predicted in zip(
        y_test[:10], predictions[:10]
    ):
        print(f"{actual:.0f}\t{predicted:.2f}")

    model.plot_predictions(
        X_test, y_test, limit=96
    )


if __name__ == "__main__":
    main()