
import heapq
import math

from routing.top_five_routes import load_graph
from routing.astar_traffic import (
    geographic_distance,
    load_coordinates
)


def longest_edge(graph, coordinates):

    longest = 0.0

    for start, destination in graph.edges():

        distance = geographic_distance(
            start,
            destination,
            coordinates
        )

        if distance > longest:
            longest = distance

    return longest


def heuristic(node, destination, coordinates, l_max):

    if node == destination or l_max == 0:
        return 0

    distance = geographic_distance(
        node,
        destination,
        coordinates
    )

    return math.ceil(distance / l_max)


def cus2(graph, coordinates, origin, destination):

    if origin not in graph or destination not in graph:
        return {
            "goal": None,
            "nodes_created": 0,
            "path": None,
            "travel_time_seconds": None,
            "steps": None
        }

    l_max = longest_edge(
        graph,
        coordinates
    )

    frontier = []

    start_h = heuristic(
        origin,
        destination,
        coordinates,
        l_max
    )

    heapq.heappush(
        frontier,
        (
            start_h,
            origin,
            0,
            [origin]
        )
    )

    nodes_created = 1
    best_g = {origin: 0}
    explored = set()

    while frontier:

        f, current, g, path = heapq.heappop(
            frontier
        )

        if current in explored:
            continue

        if current == destination:

            travel_time = 0.0

            for start, end in zip(
                path[:-1],
                path[1:]
            ):
                travel_time += graph[start][end][
                    "travel_time_seconds"
                ]

            return {
                "goal": current,
                "nodes_created": nodes_created,
                "path": path,
                "travel_time_seconds": travel_time,
                "steps": g
            }

        explored.add(current)

        for neighbor in sorted(graph.successors(current)):

            new_g = g + 1

            if neighbor in explored:
                continue

            if (
                neighbor in best_g
                and best_g[neighbor] <= new_g
            ):
                continue

            best_g[neighbor] = new_g
            nodes_created += 1

            new_h = heuristic(
                neighbor,
                destination,
                coordinates,
                l_max
            )

            heapq.heappush(
                frontier,
                (
                    new_g + new_h,
                    neighbor,
                    new_g,
                    path + [neighbor]
                )
            )

    return {
        "goal": None,
        "nodes_created": nodes_created,
        "path": None,
        "travel_time_seconds": None,
        "steps": None
    }


def main():

    graph = load_graph(use_traffic=True)
    coordinates = load_coordinates()

    result = cus2(
        graph,
        coordinates,
        2000,
        3002
    )

    print("CUS2 Modified A* Search")
    print("Goal:", result["goal"])
    print("Nodes created:", result["nodes_created"])
    print("Path:", result["path"])
    print("Steps:", result["steps"])

    if result["travel_time_seconds"] is not None:
        print(
            "Estimated travel time:",
            round(result["travel_time_seconds"], 2),
            "seconds"
        )


if __name__ == "__main__":
    main()

