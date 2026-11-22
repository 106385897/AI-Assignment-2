
import unittest

from top_five_routes import load_graph, find_top_five_routes


class TestTopFiveRoutes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = load_graph()

    def test_graph_has_40_sites(self):
        self.assertEqual(
            self.graph.number_of_nodes(),
            40
        )

    def test_main_route(self):
        routes = find_top_five_routes(
            self.graph, 2000, 3002
        )
        self.assertEqual(len(routes), 5)

    def test_different_route(self):
        routes = find_top_five_routes(
            self.graph, 3001, 3002
        )
        self.assertEqual(len(routes), 5)

    def test_reverse_route(self):
        routes = find_top_five_routes(
            self.graph, 3002, 2000
        )
        self.assertEqual(len(routes), 5)

    def test_invalid_start(self):
        routes = find_top_five_routes(
            self.graph, 9999, 3002
        )
        self.assertEqual(routes, [])

    def test_invalid_destination(self):
        routes = find_top_five_routes(
            self.graph, 2000, 9999
        )
        self.assertEqual(routes, [])

    def test_same_start_destination(self):
        routes = find_top_five_routes(
            self.graph, 2000, 2000
        )
        self.assertEqual(routes[0]["route"], [2000])
        self.assertEqual(routes[0]["distance_km"], 0)

    def test_routes_are_sorted(self):
        routes = find_top_five_routes(
            self.graph, 2000, 3002
        )
        distances = [
            result["distance_km"]
            for result in routes
        ]
        self.assertEqual(
            distances,
            sorted(distances)
        )

    def test_no_repeated_sites(self):
        routes = find_top_five_routes(
            self.graph, 2000, 3002
        )
        for result in routes:
            route = result["route"]
            self.assertEqual(
                len(route),
                len(set(route))
            )

    def test_all_routes_have_correct_endpoints(self):
        routes = find_top_five_routes(
            self.graph, 2000, 3002
        )
        for result in routes:
            self.assertEqual(result["route"][0], 2000)
            self.assertEqual(result["route"][-1], 3002)


if __name__ == "__main__":
    unittest.main(verbosity=2)

