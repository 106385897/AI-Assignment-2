
import heapq
from itertools import count

from routing.top_five_routes import load_graph
from routing.astar_traffic import geographic_distance, load_coordinates


def gbfs(graph, coordinates, origin, destination):

    if origin not in graph or destination not in graph:
        return {
            "goal": None,
            "nodes_created": 0,
            "path": None,
            "travel_time_seconds": None
        }

    frontier = []
    insertion_order = count()

    start_h = geographic_distance(
        origin,
        destination,
        coordinates
    )

    heapq.heappush(
        frontier,
        (start_h, origin, next(insertion_order), [origin])
    )

    nodes_created = 1

    while frontier:

        current_h, current, order, path = heapq.heappop(
            frontier
        )

        if current == destination:

            travel_time = sum(
                graph[path[i]][path[i + 1]]["travel_time_seconds"]
                for i in range(len(path) - 1)
            )

            return {
                "goal": current,
                "nodes_created": nodes_created,
                "path": path,
                "travel_time_seconds": travel_time
            }

        for neighbor in sorted(graph.successors(current)):

            if neighbor in path:
                continue

            new_path = path + [neighbor]

            new_h = geographic_distance(
                neighbor,
                destination,
                coordinates
            )

            heapq.heappush(
                frontier,
                (
                    new_h,
                    neighbor,
                    next(insertion_order),
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

    result = gbfs(
        graph,
        coordinates,
        2000,
        3002
    )

    print("GBFS Traffic-Aware Search")
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
