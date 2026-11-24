
import pandas as pd


def main():

    print("Loading Random Forest predictions...")

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

    flows.to_csv(
        "data/scats_average_predicted_flows.csv",
        index=False
    )

    print("\nSCATS sites:", len(flows))

    print("\nFirst 10 results:")
    print(flows.head(10).to_string(index=False))

    print(
        "\nSites above 1500 vehicles/hour:",
        (flows["RepresentativeFlowHourly"] > 1500).sum()
    )

    print("\nSaved to data/scats_average_predicted_flows.csv")


if __name__ == "__main__":
    main()
