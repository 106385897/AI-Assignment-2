
from routing.top_five_routes import load_graph


def dfs(graph, origin, destination):

    if origin not in graph or destination not in graph:
        return {
            "goal": None,
            "path": None,
            "nodes_created": 0,
            "travel_time_seconds": None
        }

    stack = [
        (origin, [origin], 0.0)
    ]

    nodes_created = 1

    while stack:

        current, path, travel_time = stack.pop()

        if current == destination:
            return {
                "goal": current,
                "path": path,
                "nodes_created": nodes_created,
                "travel_time_seconds": travel_time
            }

        for neighbor in sorted(
            graph.successors(current),
            reverse=True
        ):

            if neighbor in path:
                continue

            edge_time = graph[current][neighbor][
                "travel_time_seconds"
            ]

            stack.append(
                (
                    neighbor,
                    path + [neighbor],
                    travel_time + edge_time
                )
            )

            nodes_created += 1

    return {
        "goal": None,
        "path": None,
        "nodes_created": nodes_created,
        "travel_time_seconds": None
    }


def main():

    graph = load_graph(use_traffic=True)

    result = dfs(
        graph,
        2000,
        3002
    )

    print("DFS Traffic-Aware Search")
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
