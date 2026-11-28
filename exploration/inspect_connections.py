import pandas as pd
import math


FILE_PATH = "data/scats_sites_complete.csv"


def calculate_distance(lat1, lon1, lat2, lon2):

    radius = 6371

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(a))


def main():

    sites = pd.read_csv(FILE_PATH)

    connections = []

    for i in range(len(sites)):

        start = sites.iloc[i]

        for j in range(i + 1, len(sites)):

            end = sites.iloc[j]

            distance = calculate_distance(
                start["Latitude"],
                start["Longitude"],
                end["Latitude"],
                end["Longitude"]
            )

            if distance <= 1.5:

                connections.append({
                    "From": int(start["SCATS Number"]),
                    "To": int(end["SCATS Number"]),
                    "From Description": start["Description"],
                    "To Description": end["Description"],
                    "Straight Distance KM": round(distance, 3)
                })

    result = pd.DataFrame(connections)

    result = result.sort_values(
        ["From", "Straight Distance KM"]
    )

    print("Potential nearby connections:", len(result))

    print("\nFirst 30 candidates:")
    print(result.head(30).to_string(index=False))

    result.to_csv(
        "data/scats_connection_candidates.csv",
        index=False
    )

    print("\nSaved to data/scats_connection_candidates.csv")


if __name__ == "__main__":
    main()