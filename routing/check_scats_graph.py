
import pandas as pd
import networkx as nx


def main():

    print("Loading SCATS graph data...")

    sites = pd.read_csv(
        "data/scats_osm_nodes.csv"
    )

    connections = pd.read_csv(
        "data/scats_osm_connections.csv"
    )

    graph = nx.DiGraph()

    for site in sites["SCATS Number"]:
        graph.add_node(int(site))

    for _, edge in connections.iterrows():

        graph.add_edge(
            int(edge["From"]),
            int(edge["To"]),
            weight=float(edge["DistanceKM"])
        )

    print("\nTotal SCATS sites:", graph.number_of_nodes())
    print("Total directed connections:", graph.number_of_edges())

    print(
        "Weakly connected components:",
        nx.number_weakly_connected_components(graph)
    )

    print(
        "Strongly connected components:",
        nx.number_strongly_connected_components(graph)
    )

    start = 2000
    end = 3002

    print(f"\nChecking SCATS {start} -> {end}...")

    if nx.has_path(graph, start, end):

        route = nx.shortest_path(
            graph,
            source=start,
            target=end,
            weight="weight"
        )

        distance = nx.shortest_path_length(
            graph,
            source=start,
            target=end,
            weight="weight"
        )

        print("Route found!")
        print("SCATS route:", " -> ".join(map(str, route)))
        print(f"Estimated distance: {distance:.3f} km")

    else:

        print("No route found in candidate SCATS graph.")

    reachable = nx.descendants(graph, start)

    print(
        f"\nSCATS sites reachable from {start}:",
        len(reachable) + 1,
        "out of",
        graph.number_of_nodes()
    )


if __name__ == "__main__":
    main()
