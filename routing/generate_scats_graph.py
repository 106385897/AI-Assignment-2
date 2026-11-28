
import osmnx as ox
import networkx as nx
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


MAX_STRAIGHT_DISTANCE_KM = 2.0


def haversine(lat1, lon1, lat2, lon2):

    radius = 6371.0

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    return 2 * radius * atan2(sqrt(a), sqrt(1 - a))


def main():

    print("Loading saved road network...")

    graph = ox.load_graphml(
        "data/boroondara_road_network.graphml"
    )

    sites = pd.read_csv(
        "data/scats_osm_nodes.csv"
    )

    connections = []

    for _, start in sites.iterrows():

        for _, end in sites.iterrows():

            if start["SCATS Number"] == end["SCATS Number"]:
                continue

            straight_distance = haversine(
                start["Latitude"],
                start["Longitude"],
                end["Latitude"],
                end["Longitude"]
            )

            if straight_distance > MAX_STRAIGHT_DISTANCE_KM:
                continue

            source = int(start["OSM_Node"])
            destination = int(end["OSM_Node"])

            try:

                driving_distance = nx.shortest_path_length(
                    graph,
                    source=source,
                    target=destination,
                    weight="length"
                )

                connections.append({
                    "From": int(start["SCATS Number"]),
                    "To": int(end["SCATS Number"]),
                    "DistanceKM": round(driving_distance / 1000, 3),
                    "Source": "OpenStreetMap",
                    "Verified": "No"
                })

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    result = pd.DataFrame(connections)

    result.to_csv(
        "data/scats_osm_connections.csv",
        index=False
    )

    print("\nGenerated directed connections:", len(result))

    print("\nFirst 10 connections:")
    print(result.head(10).to_string(index=False))

    print("\nSaved to data/scats_osm_connections.csv")


if __name__ == "__main__":
    main()
