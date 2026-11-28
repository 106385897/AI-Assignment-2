import numpy as np

from models.random_forest import RandomForestTraffic
from train_random_forest import load_data, prepare_data


def main():

    print("Testing saved Random Forest model...")

    model = RandomForestTraffic()
    model.load_model()

    traffic = load_data()

    X_train, X_test, y_train, y_test = prepare_data(traffic)

    predictions = model.predict(X_test[:10])

    print("\nActual\tPredicted")

    for actual, predicted in zip(y_test[:10], predictions):
        print(f"{actual:.0f}\t{predicted:.2f}")

    assert len(predictions) == 10
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= 0)

    results = model.evaluate(X_test, y_test)

    assert np.isclose(results["MAE"], 24.7762, atol=0.0001)

    print("\nPASS: Saved model loaded successfully.")
    print("PASS: Correct number of predictions.")
    print("PASS: All predictions are finite.")
    print("PASS: No negative traffic predictions.")
    print("PASS: Original evaluation result reproduced.")


if __name__ == "__main__":
    main()