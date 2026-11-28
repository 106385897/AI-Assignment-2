
import pandas as pd


def main():

    candidates = pd.read_csv(
        "data/scats_connection_candidates.csv"
    )

    existing = pd.read_csv(
        "data/scats_edges.csv"
    )

    review = candidates.rename(columns={
        "From Description": "FromDescription",
        "To Description": "ToDescription",
        "Straight Distance KM": "StraightDistanceKM"
    }).copy()

    review["Road"] = ""
    review["DistanceKM"] = float("nan")
    review["Verified"] = "No"

    for _, edge in existing.iterrows():

        match = (
            (review["From"] == edge["From"])
            & (review["To"] == edge["To"])
        )

        if match.any():

            review.loc[match, "Road"] = str(edge["Road"])

            if pd.notna(edge["DistanceKM"]):
                review.loc[match, "DistanceKM"] = float(
                    edge["DistanceKM"]
                )

            review.loc[match, "Verified"] = str(
                edge["Verified"]
            )

    review = review[
        [
            "From",
            "To",
            "FromDescription",
            "ToDescription",
            "StraightDistanceKM",
            "Road",
            "DistanceKM",
            "Verified"
        ]
    ]

    review.to_csv(
        "data/scats_edge_review.csv",
        index=False
    )

    print("Total candidate connections:", len(review))

    print(
        "Verified connections:",
        (review["Verified"] == "Yes").sum()
    )

    print("\nSaved to data/scats_edge_review.csv")


if __name__ == "__main__":
    main()
