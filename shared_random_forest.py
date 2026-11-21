import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from data_processing import prepare_dataset
from models.random_forest import RandomForestTraffic


def main():

    print("Loading shared SCATS dataset...")

    data = prepare_dataset()

    print(f"Training samples: {len(data.X_train)}")
    print(f"Validation samples: {len(data.X_val)}")
    print(f"Testing samples: {len(data.X_test)}")

    model = RandomForestTraffic(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )

    model.train(data.X_train, data.y_train)

    # Validate using the shared validation set
    val_predictions = model.predict(data.X_val)

    val_actual = data.inverse_transform(
        data.y_val, data.site_val
    )

    val_predicted = data.inverse_transform(
        val_predictions, data.site_val
    )

    val_mae = mean_absolute_error(
        val_actual, val_predicted
    )

    print(f"\nValidation MAE: {val_mae:.4f}")

    # Evaluate on the shared test set
    predictions = model.predict(data.X_test)

    actual = data.inverse_transform(
        data.y_test, data.site_test
    )

    predicted = data.inverse_transform(
        predictions, data.site_test
    )

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    mask = actual >= 10

    mape = np.mean(
        np.abs(
            (actual[mask] - predicted[mask])
            / actual[mask]
        )
    ) * 100

    print("\nRandom Forest - Shared Test Results")
    print("----------------------------------")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2:   {r2:.4f}")

    Path("outputs").mkdir(exist_ok=True)

    model.save_model(
        "outputs/random_forest_shared.joblib"
    )

    results = pd.DataFrame({
        "SCATS": data.site_test,
        "Actual": actual,
        "Predicted": predicted
    })

    results.to_csv(
        "outputs/random_forest_predictions.csv",
        index=False
    )

    print("\nPredictions saved successfully.")


if __name__ == "__main__":
    main()