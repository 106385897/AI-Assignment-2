
import unittest

from routing.top_five_routes import load_graph
from routing.astar_traffic import astar, load_coordinates
from routing.gbfs_traffic import gbfs


class TestGBFSTraffic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = load_graph(use_traffic=True)
        cls.coordinates = load_coordinates()

    def test_route_2000_to_3002(self):

        result = gbfs(
            self.graph,
            self.coordinates,
            2000,
            3002
        )

        self.assertEqual(result["goal"], 3002)
        self.assertEqual(result["path"][0], 2000)
        self.assertEqual(result["path"][-1], 3002)
        self.assertGreater(result["travel_time_seconds"], 0)

    def test_route_4030_to_4032(self):

        result = gbfs(
            self.graph,
            self.coordinates,
            4030,
            4032
        )

        self.assertEqual(result["path"], [4030, 4032])
        self.assertAlmostEqual(
            result["travel_time_seconds"],
            95.22,
            places=2
        )

    def test_gbfs_vs_astar(self):

        greedy = gbfs(
            self.graph,
            self.coordinates,
            2000,
            3002
        )

        optimal = astar(
            self.graph,
            self.coordinates,
            2000,
            3002
        )

        self.assertGreaterEqual(
            greedy["travel_time_seconds"],
            optimal["travel_time_seconds"]
        )

    def test_same_origin_destination(self):

        result = gbfs(
            self.graph,
            self.coordinates,
            2000,
            2000
        )

        self.assertEqual(result["path"], [2000])
        self.assertEqual(result["travel_time_seconds"], 0)

    def test_invalid_origin(self):

        result = gbfs(
            self.graph,
            self.coordinates,
            -1,
            3002
        )

        self.assertIsNone(result["path"])

    def test_invalid_destination(self):

        result = gbfs(
            self.graph,
            self.coordinates,
            2000,
            -1
        )

        self.assertIsNone(result["path"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
