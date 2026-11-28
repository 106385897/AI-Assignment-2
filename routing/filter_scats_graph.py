
import pandas as pd
import networkx as nx


def main():

    print("Loading SCATS connection audit...")

    audit = pd.read_csv(
        "data/scats_connection_audit.csv"
    )

    sites = pd.read_csv(
        "data/scats_osm_nodes.csv"
    )

    clean = audit[
        audit["SkipsSCATSSite"] == False
    ].copy()

    clean = clean[
        ["From", "To", "DistanceKM"]
    ]

    clean["Source"] = "OpenStreetMap"
    clean["Verified"] = "No"

    clean.to_csv(
        "data/scats_filtered_connections.csv",
        index=False
    )

    graph = nx.DiGraph()

    for site in sites["SCATS Number"]:
        graph.add_node(int(site))

    for _, edge in clean.iterrows():

        graph.add_edge(
            int(edge["From"]),
            int(edge["To"]),
            weight=float(edge["DistanceKM"])
        )

    print("\nOriginal connections:", len(audit))
    print("Flagged connections removed:", len(audit) - len(clean))
    print("Remaining candidate connections:", len(clean))

    print("\nSCATS sites:", graph.number_of_nodes())

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

    print(f"\nChecking route {start} -> {end}...")

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
        print("Route:", " -> ".join(map(str, route)))
        print(f"Estimated distance: {distance:.3f} km")

    else:
        print("No route found.")

    print("\nSaved to data/scats_filtered_connections.csv")


if __name__ == "__main__":
    main()
