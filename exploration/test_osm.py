import osmnx as ox


def main():

    print("Connecting to OpenStreetMap...")

    place = "City of Boroondara, Victoria, Australia"

    graph = ox.graph_from_place(
        place,
        network_type="drive"
    )

    print("\nOpenStreetMap connection successful!")

    print("Road network nodes:", len(graph.nodes))
    print("Road network edges:", len(graph.edges))

    print("\nRoad network downloaded successfully.")


if __name__ == "__main__":
    main()