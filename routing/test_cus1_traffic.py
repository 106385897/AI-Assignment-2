
import unittest

from routing.top_five_routes import load_graph
from routing.cus1_traffic import cus1
from routing.ucs_traffic import ucs


class TestCUS1Traffic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = load_graph(use_traffic=True)

    def test_valid_route(self):
        result = cus1(self.graph, 2000, 3002)

        self.assertEqual(result["goal"], 3002)
        self.assertEqual(result["path"][0], 2000)
        self.assertEqual(result["path"][-1], 3002)

    def test_matches_ucs_optimal_cost(self):
        cus1_result = cus1(self.graph, 2000, 3002)
        ucs_result = ucs(self.graph, 2000, 3002)

        self.assertAlmostEqual(
            cus1_result["travel_time_seconds"],
            ucs_result["travel_time_seconds"],
            places=5
        )

    def test_invalid_origin(self):
        result = cus1(self.graph, 999999, 3002)

        self.assertIsNone(result["goal"])
        self.assertIsNone(result["path"])
        self.assertEqual(result["nodes_created"], 0)

    def test_invalid_destination(self):
        result = cus1(self.graph, 2000, 999999)

        self.assertIsNone(result["goal"])
        self.assertIsNone(result["path"])

    def test_same_origin_destination(self):
        result = cus1(self.graph, 2000, 2000)

        self.assertEqual(result["path"], [2000])
        self.assertEqual(result["travel_time_seconds"], 0)
        self.assertEqual(result["nodes_created"], 1)

    def test_nodes_created(self):
        result = cus1(self.graph, 2000, 3002)

        self.assertGreater(result["nodes_created"], 0)


if __name__ == "__main__":
    unittest.main()
