
import unittest
import networkx as nx

from routing.top_five_routes import load_graph
from routing.dfs_traffic import dfs


class TestDFSTraffic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = load_graph(use_traffic=True)

    def test_valid_route(self):
        result = dfs(self.graph, 2000, 3002)

        self.assertEqual(result["goal"], 3002)
        self.assertEqual(result["path"][0], 2000)
        self.assertEqual(result["path"][-1], 3002)

    def test_route_connections(self):
        result = dfs(self.graph, 2000, 3002)

        self.assertTrue(
            nx.is_path(self.graph, result["path"])
        )

    def test_travel_time_matches_path(self):
        result = dfs(self.graph, 2000, 3002)

        expected = nx.path_weight(
            self.graph,
            result["path"],
            weight="travel_time_seconds"
        )

        self.assertAlmostEqual(
            result["travel_time_seconds"],
            expected,
            places=3
        )

    def test_no_repeated_nodes(self):
        result = dfs(self.graph, 2000, 3002)

        self.assertEqual(
            len(result["path"]),
            len(set(result["path"]))
        )

    def test_same_origin_destination(self):
        result = dfs(self.graph, 2000, 2000)

        self.assertEqual(result["path"], [2000])
        self.assertEqual(result["travel_time_seconds"], 0)

    def test_invalid_origin(self):
        result = dfs(self.graph, -1, 3002)

        self.assertIsNone(result["path"])

    def test_invalid_destination(self):
        result = dfs(self.graph, 2000, -1)

        self.assertIsNone(result["path"])

    def test_nodes_created(self):
        result = dfs(self.graph, 2000, 3002)

        self.assertGreater(result["nodes_created"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
