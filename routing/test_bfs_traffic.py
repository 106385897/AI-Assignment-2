
import unittest
import networkx as nx

from routing.top_five_routes import load_graph
from routing.bfs_traffic import bfs


class TestBFSTraffic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = load_graph(use_traffic=True)

    def test_minimum_edges(self):

        result = bfs(self.graph, 2000, 3002)

        expected = nx.shortest_path_length(
            self.graph,
            2000,
            3002
        )

        self.assertEqual(
            len(result["path"]) - 1,
            expected
        )

    def test_direct_route(self):

        result = bfs(self.graph, 4030, 4032)

        self.assertEqual(result["path"], [4030, 4032])

    def test_travel_time_matches_path(self):

        result = bfs(self.graph, 2000, 3002)

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

        result = bfs(self.graph, 2000, 3002)

        self.assertEqual(
            len(result["path"]),
            len(set(result["path"]))
        )

    def test_same_origin_destination(self):

        result = bfs(self.graph, 2000, 2000)

        self.assertEqual(result["path"], [2000])
        self.assertEqual(result["travel_time_seconds"], 0)

    def test_invalid_origin(self):

        result = bfs(self.graph, -1, 3002)

        self.assertIsNone(result["path"])

    def test_invalid_destination(self):

        result = bfs(self.graph, 2000, -1)

        self.assertIsNone(result["path"])

    def test_nodes_created(self):

        result = bfs(self.graph, 2000, 3002)

        self.assertGreater(result["nodes_created"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

