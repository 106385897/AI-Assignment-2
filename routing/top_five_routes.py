
import pandas as pd
import networkx as nx
from itertools import islice

from routing.travel_time import calculate_edge_cost


def load_graph(use_traffic=False, allow_historical_fallback=False):

    connections = pd.read_csv(
        "data/scats_filtered_connections.csv"
    )

    graph = nx.DiGraph()
    flow_lookup = {}
    flow_sources = {}

    if use_traffic:

        flows = pd.read_csv(
            "data/scats_average_predicted_flows.csv"
        )

        flow_lookup = dict(
            zip(
                flows["SCATS"].astype(int),
                flows["RepresentativeFlow15Min"]
            )
        )

        flow_sources = {
            site: "Random Forest prediction"
            for site in flow_lookup
        }

        if allow_historical_fallback:

            fallback = pd.read_csv(
                "data/scats_historical_fallback_flows.csv"
            )

            for _, row in fallback.iterrows():

                site = int(row["SCATS"])

                if site not in flow_lookup:

                    flow_lookup[site] = float(
                        row["RepresentativeFlow15Min"]
                    )

                    flow_sources[site] = row["Source"]

    for _, edge in connections.iterrows():

        start = int(edge["From"])
        destination = int(edge["To"])
        distance = float(edge["DistanceKM"])

        if not use_traffic:

            graph.add_edge(
                start,
                destination,
                distance_km=distance
            )

        else:

            flow = flow_lookup.get(start)

            if flow is None or pd.isna(flow):
                continue

            try:

                travel_time = calculate_edge_cost(
                    distance,
                    float(flow)
                )

            except ValueError:
                continue

            graph.add_edge(
                start,
                destination,
                distance_km=distance,
                travel_time_seconds=travel_time,
                flow_source=flow_sources[start]
            )

    return graph


def find_top_five_routes(
    graph,
    start,
    destination,
    use_traffic=False
):

    if start not in graph or destination not in graph:
        return []

    weight = (
        "travel_time_seconds"
        if use_traffic
        else "distance_km"
    )

    try:

        routes = list(
            islice(
                nx.shortest_simple_paths(
                    graph,
                    source=start,
                    target=destination,
                    weight=weight
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

            result = {
                "route": route,
                "distance_km": distance
            }

            if use_traffic:

                result["travel_time_seconds"] = (
                    nx.path_weight(
                        graph,
                        route,
                        weight="travel_time_seconds"
                    )
                    if len(route) > 1
                    else 0.0
                )

                result["flow_sources"] = sorted({
                    graph[route[i]][route[i + 1]]["flow_source"]
                    for i in range(len(route) - 1)
                })

            results.append(result)

        return results

    except nx.NetworkXNoPath:
        return []


def main():

    graph = load_graph(
        use_traffic=True,
        allow_historical_fallback=False
    )

    start = 2000
    destination = 3002

    print(
        f"Top five traffic-aware routes: "
        f"{start} -> {destination}"
    )

    results = find_top_five_routes(
        graph,
        start,
        destination,
        use_traffic=True
    )

    if not results:
        print("No routes found with available traffic data.")
        return

    for number, result in enumerate(results, start=1):

        print(f"\nRoute {number}:")
        print(" -> ".join(map(str, result["route"])))
        print(f"Distance: {result['distance_km']:.3f} km")

        print(
            f"Estimated travel time: "
            f"{result['travel_time_seconds']:.2f} seconds"
        )

        print(
            "Traffic source:",
            ", ".join(result["flow_sources"])
        )


if __name__ == "__main__":
    main()
