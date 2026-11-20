import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class RandomForestTraffic:

    def __init__(self, n_estimators=100, max_depth=15,
                 min_samples_split=5, random_state=42):

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )

    # Convert sequence data into the format required by Random Forest
    def prepare_input(self, X):
        X = np.asarray(X)

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        elif X.ndim == 1:
            X = X.reshape(1, -1)

        return X

    # Train the model using historical traffic data
    def train(self, X_train, y_train):
        X_train = self.prepare_input(X_train)
        y_train = np.asarray(y_train).ravel()

        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        print("Training completed successfully.")

    # Predict future traffic volume
    def predict(self, X):
        X = self.prepare_input(X)
        predictions = self.model.predict(X)

        return np.maximum(predictions, 0)

    # Evaluate the performance of the model
    def evaluate(self, X_test, y_test):
        y_test = np.asarray(y_test).ravel()
        predictions = self.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        results = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

        print("\nRandom Forest Evaluation")
        print("-------------------------")
        print(f"MAE:  {mae:.4f}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")

        return results

    # Visualise actual and predicted traffic volumes
    def plot_predictions(self, X_test, y_test, limit=200):
        y_test = np.asarray(y_test).ravel()
        predictions = self.predict(X_test)

        plt.figure(figsize=(12, 5))

        plt.plot(y_test[:limit], label="Actual Traffic")
        plt.plot(predictions[:limit], label="Predicted Traffic")

        plt.title("Random Forest - Actual vs Predicted Traffic")
        plt.xlabel("Test observation")
        plt.ylabel("Traffic volume")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Save the trained model for future use
    def save_model(self, filename="saved_models/random_forest.joblib"):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")

    # Load a previously trained model
    def load_model(self, filename="saved_models/random_forest.joblib"):
        self.model = joblib.load(filename)
        print("Random Forest model loaded successfully.")