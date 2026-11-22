
import osmnx as ox
import networkx as nx
import pandas as pd


def main():

    print("Loading saved road network...")

    graph = ox.load_graphml(
        "data/boroondara_road_network.graphml"
    )

    sites = pd.read_csv(
        "data/scats_osm_nodes.csv"
    )

    connections = pd.read_csv(
        "data/scats_osm_connections.csv"
    )

    node_to_site = dict(
        zip(
            sites["OSM_Node"].astype(int),
            sites["SCATS Number"].astype(int)
        )
    )

    site_to_node = dict(
        zip(
            sites["SCATS Number"].astype(int),
            sites["OSM_Node"].astype(int)
        )
    )

    results = []

    for _, edge in connections.iterrows():

        start = int(edge["From"])
        end = int(edge["To"])

        source = site_to_node[start]
        destination = site_to_node[end]

        try:
            path = nx.shortest_path(
                graph,
                source=source,
                target=destination,
                weight="length"
            )

            intermediate_sites = []

            for node in path[1:-1]:

                if node in node_to_site:
                    intermediate_sites.append(
                        node_to_site[node]
                    )

            intermediate_sites = list(
                dict.fromkeys(intermediate_sites)
            )

            results.append({
                "From": start,
                "To": end,
                "DistanceKM": edge["DistanceKM"],
                "IntermediateSites": ",".join(
                    map(str, intermediate_sites)
                ),
                "SkipsSCATSSite": len(intermediate_sites) > 0
            })

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    result = pd.DataFrame(results)

    result.to_csv(
        "data/scats_connection_audit.csv",
        index=False
    )

    skipped = result[
        result["SkipsSCATSSite"]
    ]

    print("\nTotal connections checked:", len(result))
    print("Connections passing through another SCATS node:", len(skipped))

    print("\nExamples:")

    if skipped.empty:
        print("No intermediate SCATS nodes detected.")
    else:
        print(
            skipped.head(10).to_string(index=False)
        )

    print("\nSaved to data/scats_connection_audit.csv")


if __name__ == "__main__":
    main()
