
import osmnx as ox
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def main():

    sites = pd.read_csv(
        "data/scats_osm_nodes.csv"
    )

    print("Loading road network...")

    graph = ox.graph_from_place(
        "City of Boroondara, Victoria, Australia",
        network_type="drive"
    )

    start = 4030
    end = 4032

    start_site = sites.loc[
        sites["SCATS Number"] == start
    ].iloc[0]

    end_site = sites.loc[
        sites["SCATS Number"] == end
    ].iloc[0]

    source = int(start_site["OSM_Node"])
    destination = int(end_site["OSM_Node"])

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

    print(f"\nSCATS {start} -> {end}")
    print(f"OSM driving distance: {distance:.0f} metres")

    for site, node in [
        (start_site, source),
        (end_site, destination)
    ]:

        snap_distance = ox.distance.great_circle(
            site["Latitude"],
            site["Longitude"],
            graph.nodes[node]["y"],
            graph.nodes[node]["x"]
        )

        print(
            f"SCATS {int(site['SCATS Number'])} "
            f"snapping distance: {snap_distance:.1f} metres"
        )

    fig, ax = ox.plot_graph_route(
        graph,
        route,
        route_color="red",
        route_linewidth=5,
        node_size=0,
        show=False,
        close=False
    )

    route_x = [
        graph.nodes[node]["x"]
        for node in route
    ]

    route_y = [
        graph.nodes[node]["y"]
        for node in route
    ]

    margin = 0.003

    ax.set_xlim(
        min(route_x) - margin,
        max(route_x) + margin
    )

    ax.set_ylim(
        min(route_y) - margin,
        max(route_y) + margin
    )

    ax.set_title(
        f"SCATS {start} to {end} - {distance:.0f} m"
    )

    plt.show()


if __name__ == "__main__":
    main()
