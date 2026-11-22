
import pandas as pd
import networkx as nx
from itertools import islice


def main():

    print("Loading filtered SCATS connections...")

    connections = pd.read_csv(
        "data/scats_filtered_connections.csv"
    )

    graph = nx.DiGraph()

    for _, edge in connections.iterrows():

        graph.add_edge(
            int(edge["From"]),
            int(edge["To"]),
            weight=float(edge["DistanceKM"])
        )

    start = 2000
    destination = 3002

    print(f"\nFinding top five routes: {start} -> {destination}")

    try:

        routes = list(
            islice(
                nx.shortest_simple_paths(
                    graph,
                    source=start,
                    target=destination,
                    weight="weight"
                ),
                5
            )
        )

        print("\nTOP FIVE ROUTES")

        for number, route in enumerate(routes, start=1):

            distance = nx.path_weight(
                graph,
                route,
                weight="weight"
            )

            print(f"\nRoute {number}:")
            print(" -> ".join(map(str, route)))
            print(f"Distance: {distance:.3f} km")

        print(f"\nTotal routes found: {len(routes)}")

    except (nx.NetworkXNoPath, nx.NodeNotFound):

        print("\nNo route found.")


if __name__ == "__main__":
    main()
