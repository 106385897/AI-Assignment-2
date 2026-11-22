
from routing.top_five_routes import load_graph, find_top_five_routes


def get_routes(start, destination):

    graph = load_graph()

    start = int(start)
    destination = int(destination)

    routes = find_top_five_routes(
        graph,
        start,
        destination
    )

    results = []

    for number, route in enumerate(routes, start=1):

        results.append({
            "route_number": number,
            "scats_sites": route["route"],
            "distance_km": round(route["distance_km"], 3),
            "travel_time_seconds": None,
            "status": "Distance-only prototype"
        })

    return results


if __name__ == "__main__":

    routes = get_routes(2000, 3002)

    for route in routes:
        print(route)
