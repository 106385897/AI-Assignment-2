
from pathlib import Path

import osmnx as ox
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from routing.top_five_routes import (
    load_graph,
    find_top_five_routes
)


def main():

    start_scats = 2000
    end_scats = 3002

    print("Loading traffic-aware SCATS routes...")

    scats_graph = load_graph(use_traffic=True)

    routes = find_top_five_routes(
        scats_graph,
        start_scats,
        end_scats,
        use_traffic=True
    )

    if not routes:
        print("No traffic-aware routes found.")
        return

    sites = pd.read_csv(
        "data/scats_osm_nodes.csv"
    )

    node_lookup = dict(
        zip(
            sites["SCATS Number"].astype(int),
            sites["OSM_Node"].astype(int)
        )
    )

    print("Loading OpenStreetMap road network...")

    osm_graph = ox.graph_from_place(
        "City of Boroondara, Victoria, Australia",
        network_type="drive"
    )

    colors = [
        "#ff4040",
        "#3498db",
        "#2ecc71",
        "#b85cff",
        "#ff9900"
    ]

    fig, ax = ox.plot_graph(
        osm_graph,
        node_size=0,
        edge_color="#777777",
        edge_linewidth=0.5,
        bgcolor="#111111",
        figsize=(12, 9),
        show=False,
        close=False
    )

    all_x = []
    all_y = []
    legend_items = []

    successful_routes = 0

    for i, result in enumerate(routes):

        scats_route = result["route"]
        full_osm_route = []

        missing_sites = [
            site for site in scats_route
            if site not in node_lookup
        ]

        if missing_sites:
            print(
                f"Route {i + 1}: "
                f"Missing SCATS mappings {missing_sites}"
            )
            continue

        for start, end in zip(
            scats_route[:-1],
            scats_route[1:]
        ):

            source = node_lookup[start]
            destination = node_lookup[end]

            try:
                segment = nx.shortest_path(
                    osm_graph,
                    source=source,
                    target=destination,
                    weight="length"
                )

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                print(
                    f"Route {i + 1}: "
                    f"No OSM road connection between "
                    f"SCATS {start} and {end}"
                )
                full_osm_route = []
                break

            if full_osm_route:
                full_osm_route.extend(
                    segment[1:]
                )
            else:
                full_osm_route.extend(
                    segment
                )

        if not full_osm_route:
            continue

        fig, ax = ox.plot_graph_route(
            osm_graph,
            full_osm_route,
            route_color=colors[i],
            route_linewidth=3 + (len(routes) - i) * 0.3,
            node_size=0,
            ax=ax,
            show=False,
            close=False
        )

        all_x.extend(
            osm_graph.nodes[node]["x"]
            for node in full_osm_route
        )

        all_y.extend(
            osm_graph.nodes[node]["y"]
            for node in full_osm_route
        )

        travel_time = result["travel_time_seconds"]

        minutes = int(travel_time // 60)
        seconds = int(round(travel_time % 60))

        if seconds == 60:
            minutes += 1
            seconds = 0

        legend_items.append(
            Line2D(
                [0],
                [0],
                color=colors[i],
                linewidth=3,
                label=(
                    f"Route {i + 1}: "
                    f"{minutes}m {seconds:02d}s"
                )
            )
        )

        successful_routes += 1

        print(
            f"Route {i + 1}: "
            f"{travel_time:.2f} seconds"
        )

    if successful_routes == 0:
        print("No routes could be displayed.")
        plt.close(fig)
        return

    start_node = node_lookup[start_scats]
    end_node = node_lookup[end_scats]

    start_x = osm_graph.nodes[start_node]["x"]
    start_y = osm_graph.nodes[start_node]["y"]

    end_x = osm_graph.nodes[end_node]["x"]
    end_y = osm_graph.nodes[end_node]["y"]

    ax.scatter(
        start_x,
        start_y,
        s=130,
        color="lime",
        edgecolor="white",
        linewidth=1.5,
        zorder=10
    )

    ax.scatter(
        end_x,
        end_y,
        s=130,
        color="red",
        edgecolor="white",
        linewidth=1.5,
        zorder=10
    )

    legend_items.extend([
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="lime",
            markeredgecolor="white",
            markersize=9,
            label=f"Start: SCATS {start_scats}"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="red",
            markeredgecolor="white",
            markersize=9,
            label=f"Destination: SCATS {end_scats}"
        )
    ])

    if all_x and all_y:

        margin = 0.004

        ax.set_xlim(
            min(all_x) - margin,
            max(all_x) + margin
        )

        ax.set_ylim(
            min(all_y) - margin,
            max(all_y) + margin
        )

    ax.set_title(
        "Top Five Traffic-Aware Routes\n"
        f"SCATS {start_scats} to {end_scats}",
        fontsize=16,
        color="black",
        pad=15
    )

    ax.legend(
        handles=legend_items,
        loc="lower left",
        fontsize=9,
        facecolor="#222222",
        edgecolor="white",
        labelcolor="white"
    )

    output_path = Path(
        "outputs/top_five_osm_routes.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    print(
        f"\nSuccessfully visualised "
        f"{successful_routes} routes."
    )

    print(
        f"Map saved to {output_path}"
    )

    print(
        "\nNote: OSM roads are used for "
        "approximate visualisation."
    )

    print(
        "Route rankings and travel times "
        "come from the SCATS traffic model."
    )

    plt.show()


if __name__ == "__main__":
    main()
