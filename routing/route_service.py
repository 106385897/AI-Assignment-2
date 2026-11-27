
from routing.top_five_routes import (
    load_graph,
    find_top_five_routes
)


def get_routes(start, destination, use_traffic=True):

    start = int(start)
    destination = int(destination)

    graph = load_graph(
    use_traffic=use_traffic,
    allow_historical_fallback=use_traffic
    )

    routes = find_top_five_routes(
        graph,
        start,
        destination,
        use_traffic=use_traffic
    )

    results = []

    for number, route in enumerate(routes, start=1):

        result = {
            "route_number": number,
            "scats_sites": route["route"],
            "distance_km": round(
                route["distance_km"], 3
            ),
            "travel_time_seconds": (
                round(route["travel_time_seconds"], 2)
                if use_traffic else None
            ),
            "status": (
                "Traffic-aware"
                if use_traffic else "Distance-only prototype"
            )
        }

        if use_traffic:
            result["flow_sources"] = route.get(
                "flow_sources", []
            )

        results.append(result)

    return results


if __name__ == "__main__":

    routes = get_routes(
        2000,
        3002,
        use_traffic=True
    )

    for route in routes:
        print(route)
