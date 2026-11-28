
import heapq
from itertools import count

from routing.top_five_routes import load_graph


def ucs(graph, start, destination):

    if start not in graph or destination not in graph:
        return {
            "goal": None,
            "nodes_created": 0,
            "path": None,
            "travel_time_seconds": None
        }

    insertion_order = count()

    frontier = [
        (0.0, start, next(insertion_order), [start])
    ]

    best_cost = {start: 0.0}
    nodes_created = 1

    while frontier:

        current_cost, current, order, path = heapq.heappop(
            frontier
        )

        if current_cost > best_cost.get(
            current,
            float("inf")
        ):
            continue

        if current == destination:
            return {
                "goal": current,
                "nodes_created": nodes_created,
                "path": path,
                "travel_time_seconds": current_cost
            }

        for neighbor in sorted(graph.successors(current)):

            edge_cost = graph[current][neighbor][
                "travel_time_seconds"
            ]

            new_cost = current_cost + edge_cost

            if new_cost < best_cost.get(
                neighbor,
                float("inf")
            ):

                best_cost[neighbor] = new_cost
                new_path = path + [neighbor]

                heapq.heappush(
                    frontier,
                    (
                        new_cost,
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

    result = ucs(
        graph,
        2000,
        3002
    )

    print("UCS Traffic-Aware Search")
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
