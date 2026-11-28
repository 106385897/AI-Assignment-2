
import heapq
import math
import pandas as pd
from itertools import count

from routing.top_five_routes import load_graph


def geographic_distance(site1, site2, coordinates):

    lat1, lon1 = coordinates[site1]
    lat2, lon2 = coordinates[site2]

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    return 6371 * 2 * math.asin(math.sqrt(a))


def load_coordinates():

    data = pd.read_csv(
        "data/scats_sites_complete.csv"
    )

    return {
        int(row["SCATS Number"]): (
            float(row["Latitude"]),
            float(row["Longitude"])
        )
        for _, row in data.iterrows()
    }


def heuristic_scale(graph, coordinates):

    scale = float("inf")

    for start, destination, edge in graph.edges(data=True):

        distance = geographic_distance(
            start,
            destination,
            coordinates
        )

        if distance > 0:

            cost = edge["travel_time_seconds"]

            scale = min(
                scale,
                cost / distance
            )

    if scale == float("inf"):
        return 0.0

    return scale


def heuristic(node, destination, coordinates, scale):

    distance = geographic_distance(
        node,
        destination,
        coordinates
    )

    return distance * scale


def astar(graph, coordinates, origin, destination):

    if origin not in graph or destination not in graph:
        return {
            "goal": None,
            "nodes_created": 0,
            "path": None,
            "travel_time_seconds": None
        }

    scale = heuristic_scale(
        graph,
        coordinates
    )

    frontier = []
    insertion_order = count()

    start_h = heuristic(
        origin,
        destination,
        coordinates,
        scale
    )

    heapq.heappush(
        frontier,
        (
            start_h,
            origin,
            next(insertion_order),
            0.0,
            [origin]
        )
    )

    nodes_created = 1
    best_cost = {origin: 0.0}

    while frontier:

        current_f, current, order, current_g, path = (
            heapq.heappop(frontier)
        )

        if current_g > best_cost.get(
            current,
            float("inf")
        ):
            continue

        if current == destination:
            return {
                "goal": current,
                "nodes_created": nodes_created,
                "path": path,
                "travel_time_seconds": current_g
            }

        for neighbor in sorted(graph.successors(current)):

            edge_cost = graph[current][neighbor][
                "travel_time_seconds"
            ]

            new_g = current_g + edge_cost

            if new_g < best_cost.get(
                neighbor,
                float("inf")
            ):

                best_cost[neighbor] = new_g

                new_h = heuristic(
                    neighbor,
                    destination,
                    coordinates,
                    scale
                )

                new_f = new_g + new_h
                new_path = path + [neighbor]

                heapq.heappush(
                    frontier,
                    (
                        new_f,
                        neighbor,
                        next(insertion_order),
                        new_g,
                        new_path
                    )
                )

                nodes_created += 1

    return {
        "goal": None,
        "nodes_created": nodes_created,
        "path": None,
        "travel_time_seconds": None
    }


def main():

    graph = load_graph(use_traffic=True)
    coordinates = load_coordinates()

    result = astar(
        graph,
        coordinates,
        2000,
        3002
    )

    print("A* Traffic-Aware Search")
    print("Goal:", result["goal"])
    print("Nodes created:", result["nodes_created"])
    print("Path:", result["path"])

    if result["travel_time_seconds"] is not None:
        print(
            "Estimated travel time:",
            round(result["travel_time_seconds"], 2),
            "seconds"
        )


if __name__ == "__main__":
    main()
