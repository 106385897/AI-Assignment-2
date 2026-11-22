
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

    node_lookup = dict(
        zip(sites["SCATS Number"], sites["OSM_Node"])
    )

    start = 2000
    end = 3002

    source = int(node_lookup[start])
    destination = int(node_lookup[end])

    print("\nChecking assignment route...")
    print(f"Start: SCATS {start}")
    print(f"Destination: SCATS {end}")

    try:

        route = nx.shortest_path(
            graph,
            source=source,
            target=destination,
            weight="length"
        )

        distance = nx.shortest_path_length(
            graph,
            source=source,
            target=destination,
            weight="length"
        )

        print("\nRoute found successfully!")
        print(f"Driving distance: {distance / 1000:.3f} km")
        print("Road nodes in route:", len(route))

    except (nx.NetworkXNoPath, nx.NodeNotFound):

        print("\nNo driving route found.")

    print("\nChecking network connectivity...")

    connected = 0

    for site in sites["OSM_Node"]:

        if nx.has_path(graph, source, int(site)):
            connected += 1

    print(f"SCATS sites reachable from {start}: {connected}/40")


if __name__ == "__main__":
    main()
