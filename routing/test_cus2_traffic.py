
import unittest

import networkx as nx

from routing.top_five_routes import load_graph
from routing.astar_traffic import load_coordinates
from routing.cus2_traffic import cus2


class TestCUS2Traffic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = load_graph(use_traffic=True)
        cls.coordinates = load_coordinates()

    def test_valid_route(self):
        result = cus2(
            self.graph,
            self.coordinates,
            2000,
            3002
        )

        self.assertEqual(result["goal"], 3002)
        self.assertEqual(result["path"][0], 2000)
        self.assertEqual(result["path"][-1], 3002)

    def test_fewest_steps(self):
        result = cus2(
            self.graph,
            self.coordinates,
            2000,
            3002
        )

        shortest_steps = nx.shortest_path_length(
            self.graph,
            2000,
            3002
        )

        self.assertEqual(
            result["steps"],
            shortest_steps
        )

    def test_travel_time_calculation(self):
        result = cus2(
            self.graph,
            self.coordinates,
            2000,
            3002
        )

        calculated_time = sum(
            self.graph[start][end]["travel_time_seconds"]
            for start, end in zip(
                result["path"][:-1],
                result["path"][1:]
            )
        )

        self.assertAlmostEqual(
            result["travel_time_seconds"],
            calculated_time,
            places=5
        )

    def test_invalid_origin(self):
        result = cus2(
            self.graph,
            self.coordinates,
            999999,
            3002
        )

        self.assertIsNone(result["goal"])
        self.assertEqual(result["nodes_created"], 0)

    def test_invalid_destination(self):
        result = cus2(
            self.graph,
            self.coordinates,
            2000,
            999999
        )

        self.assertIsNone(result["goal"])

    def test_same_origin_destination(self):
        result = cus2(
            self.graph,
            self.coordinates,
            2000,
            2000
        )

        self.assertEqual(result["path"], [2000])
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["travel_time_seconds"], 0)

    def test_nodes_created(self):
        result = cus2(
            self.graph,
            self.coordinates,
            2000,
            3002
        )

        self.assertGreater(result["nodes_created"], 0)


if __name__ == "__main__":
    unittest.main()
