
import pandas as pd

from top_five_routes import load_graph, find_top_five_routes


def main():

    graph = load_graph()

    flows = pd.read_csv(
        "data/scats_average_predicted_flows.csv"
    )

    flow_lookup = dict(
        zip(
            flows["SCATS"].astype(int),
            flows["PredictedFlow15Min"]
        )
    )

    routes = find_top_five_routes(
        graph,
        2000,
        3002
    )

    print("Checking traffic data for five routes...")

    for number, result in enumerate(routes, start=1):

        route = result["route"]

        missing = [
            site for site in route
            if site not in flow_lookup
        ]

        over_capacity = [
            site for site in route
            if site in flow_lookup
            and flow_lookup[site] * 4 > 1500
        ]

        print(f"\nRoute {number}:")
        print(" -> ".join(map(str, route)))
        print("Missing predictions:", missing)
        print("Sites above 1500 vehicles/hour:", over_capacity)

        if not missing and not over_capacity:
            print("Status: Ready for travel-time testing")
        else:
            print("Status: Requires further handling")


if __name__ == "__main__":
    main()
