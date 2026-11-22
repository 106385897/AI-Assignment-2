
import osmnx as ox
import pandas as pd


def main():

    print("Loading SCATS intersections...")

    sites = pd.read_csv(
        "data/scats_sites_complete.csv"
    )

    print("Total SCATS sites:", len(sites))

    print("\nDownloading Boroondara road network...")

    graph = ox.graph_from_place(
        "City of Boroondara, Victoria, Australia",
        network_type="drive"
    )

    ox.save_graphml(
        graph,
        filepath="data/boroondara_road_network.graphml"
    )

    print("Road network saved successfully.")

    print("Road nodes:", len(graph.nodes))
    print("Road edges:", len(graph.edges))

    nodes = ox.distance.nearest_nodes(
        graph,
        X=sites["Longitude"].to_numpy(),
        Y=sites["Latitude"].to_numpy()
    )

    sites["OSM_Node"] = nodes

    print("\nMatched SCATS intersections:")
    print(
        sites[
            ["SCATS Number", "Description", "OSM_Node"]
        ].to_string(index=False)
    )

    print("\nMatched sites:", sites["OSM_Node"].notna().sum())
    print("Unique matched road nodes:", sites["OSM_Node"].nunique())

    sites.to_csv(
        "data/scats_osm_nodes.csv",
        index=False
    )

    print("\nSaved to data/scats_osm_nodes.csv")


if __name__ == "__main__":
    main()
