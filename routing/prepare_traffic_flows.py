
import sys
import pandas as pd

from data_processing import prepare_dataset
from models.random_forest import RandomForestTraffic


def get_predictions_from_model():

    print("Loading shared dataset...")
    data = prepare_dataset()

    print("Loading trained Random Forest...")
    model = RandomForestTraffic()
    model.load_model("outputs/random_forest_shared.joblib")

    print("Generating traffic predictions...")
    predictions = model.predict(data.X_test)

    predicted_flows = data.inverse_transform(
        predictions,
        data.site_test
    )

    return pd.DataFrame({
        "SCATS": data.site_test,
        "Predicted": predicted_flows
    })


def main():

    use_model = "--from-model" in sys.argv

    if use_model:
        predictions = get_predictions_from_model()
    else:
        print("Loading saved Random Forest predictions...")

        predictions = pd.read_csv(
            "outputs/random_forest_predictions.csv"
        )

    raw = pd.read_excel(
        "Scats_Data_October_2006.xls",
        sheet_name="Data",
        header=1,
        engine="calamine"
    )

    raw["Date"] = pd.to_datetime(
        raw["Date"]
    ).dt.normalize()

    detector_counts = raw.groupby(
        ["SCATS Number", "Date"]
    ).size()

    usual_counts = detector_counts.groupby(
        "SCATS Number"
    ).agg(lambda values: values.mode().iloc[0])

    flows = predictions.groupby(
        "SCATS",
        as_index=False
    )["Predicted"].mean()

    flows = flows.rename(
        columns={
            "Predicted": "PredictedTotalFlow15Min"
        }
    )

    flows["DetectorCount"] = flows["SCATS"].map(
        usual_counts
    )

    flows["RepresentativeFlow15Min"] = (
        flows["PredictedTotalFlow15Min"]
        / flows["DetectorCount"]
    )

    flows["RepresentativeFlowHourly"] = (
        flows["RepresentativeFlow15Min"] * 4
    )

    if flows["DetectorCount"].isna().any():
        raise ValueError("Missing detector counts")

    if (
        flows["RepresentativeFlowHourly"] > 1500
    ).any():
        raise ValueError(
            "Some representative flows exceed road capacity"
        )

    flows.to_csv(
        "data/scats_average_predicted_flows.csv",
        index=False
    )

    print("\nSCATS sites:", len(flows))
    print("\nFirst five results:")
    print(flows.head().to_string(index=False))

    print("\nPredictions prepared for routing.")


if __name__ == "__main__":
    main()
