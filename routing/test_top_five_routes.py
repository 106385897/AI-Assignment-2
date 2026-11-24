
import unittest
import networkx as nx

from routing.top_five_routes import load_graph, find_top_five_routes


class TestTopFiveRoutes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.distance_graph = load_graph()
        cls.traffic_graph = load_graph(use_traffic=True)

    def test_graph_has_nodes(self):
        self.assertGreater(
            self.distance_graph.number_of_nodes(), 0
        )

    def test_graph_has_edges(self):
        self.assertGreater(
            self.distance_graph.number_of_edges(), 0
        )

    def test_five_distance_routes(self):
        routes = find_top_five_routes(
            self.distance_graph, 2000, 3002
        )
        self.assertEqual(len(routes), 5)

    def test_distance_routes_sorted(self):
        routes = find_top_five_routes(
            self.distance_graph, 2000, 3002
        )
        distances = [r["distance_km"] for r in routes]
        self.assertEqual(distances, sorted(distances))

    def test_five_traffic_routes(self):
        routes = find_top_five_routes(
            self.traffic_graph, 2000, 3002,
            use_traffic=True
        )
        self.assertEqual(len(routes), 5)

    def test_traffic_routes_sorted(self):
        routes = find_top_five_routes(
            self.traffic_graph, 2000, 3002,
            use_traffic=True
        )
        times = [r["travel_time_seconds"] for r in routes]
        self.assertEqual(times, sorted(times))

    def test_traffic_edge_costs_positive(self):
        for _, _, edge in self.traffic_graph.edges(data=True):
            self.assertGreater(
                edge["travel_time_seconds"], 0
            )

    def test_no_repeated_nodes(self):
        routes = find_top_five_routes(
            self.traffic_graph, 2000, 3002,
            use_traffic=True
        )
        for result in routes:
            route = result["route"]
            self.assertEqual(len(route), len(set(route)))

    def test_invalid_start(self):
        routes = find_top_five_routes(
            self.traffic_graph, -1, 3002,
            use_traffic=True
        )
        self.assertEqual(routes, [])

    def test_same_origin_destination(self):
        routes = find_top_five_routes(
            self.distance_graph, 2000, 2000
        )
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["route"], [2000])
        self.assertEqual(routes[0]["distance_km"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
