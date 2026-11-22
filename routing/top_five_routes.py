
import pandas as pd
import networkx as nx
from itertools import islice


def load_graph():

    connections = pd.read_csv(
        "data/scats_filtered_connections.csv"
    )

    graph = nx.DiGraph()

    for _, edge in connections.iterrows():

        graph.add_edge(
            int(edge["From"]),
            int(edge["To"]),
            distance_km=float(edge["DistanceKM"])
        )

    return graph


def find_top_five_routes(graph, start, destination):

    if start not in graph or destination not in graph:
        return []

    try:

        routes = list(
            islice(
                nx.shortest_simple_paths(
                    graph,
                    source=start,
                    target=destination,
                    weight="distance_km"
                ),
                5
            )
        )

        results = []

        for route in routes:

            distance = nx.path_weight(
                graph,
                route,
                weight="distance_km"
            )

            results.append({
                "route": route,
                "distance_km": distance
            })

        return results

    except nx.NetworkXNoPath:
        return []


def main():

    graph = load_graph()

    start = 2000
    destination = 3002

    print(f"Top five routes: {start} -> {destination}")

    results = find_top_five_routes(
        graph,
        start,
        destination
    )

    if not results:
        print("No routes found.")
        return

    for number, result in enumerate(results, start=1):

        print(f"\nRoute {number}:")
        print(" -> ".join(map(str, result["route"])))
        print(f"Distance: {result['distance_km']:.3f} km")


if __name__ == "__main__":
    main()
