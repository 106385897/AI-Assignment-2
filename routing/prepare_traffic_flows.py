
import pandas as pd


def main():

    print("Loading Random Forest predictions...")

    predictions = pd.read_csv(
        "outputs/random_forest_predictions.csv"
    )

    flows = predictions.groupby(
        "SCATS",
        as_index=False
    )["Predicted"].mean()

    flows = flows.rename(
        columns={
            "Predicted": "PredictedFlow15Min"
        }
    )

    flows["PredictedFlow15Min"] = (
        flows["PredictedFlow15Min"].clip(lower=0)
    )

    flows.to_csv(
        "data/scats_average_predicted_flows.csv",
        index=False
    )

    print("\nSCATS sites:", len(flows))

    print("\nFirst 10 predicted flows:")
    print(flows.head(10).to_string(index=False))

    print("\nSaved to data/scats_average_predicted_flows.csv")


if __name__ == "__main__":
    main()
