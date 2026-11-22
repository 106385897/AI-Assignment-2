from models.random_forest import RandomForestTraffic
from train_random_forest import load_data
from train_random_forest import prepare_data
from train_random_forest import compare_baseline


def main():

    print("Training final Random Forest model...")

    traffic = load_data()

    X_train, X_test, y_train, y_test = prepare_data(traffic)

    model = RandomForestTraffic(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )

    model.train(X_train, y_train)

    results = model.evaluate(X_test, y_test)

    compare_baseline(X_test, y_test, results)

    model.save_model(
        "saved_models/random_forest_final.joblib"
    )

    model.plot_predictions(
        X_test, y_test, limit=96
    )


if __name__ == "__main__":
    main()