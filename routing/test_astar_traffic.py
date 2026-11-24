
import unittest
import networkx as nx

from routing.top_five_routes import load_graph
from routing.astar_traffic import astar, load_coordinates


class TestAStarTraffic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = load_graph(use_traffic=True)
        cls.coordinates = load_coordinates()

    def test_routes_match_networkx(self):

        pairs = [
            (2000, 3002),
            (4030, 4032),
            (3001, 3002),
            (970, 2200),
            (3120, 4266),
            (2000, 4030),
            (4032, 3001),
            (2820, 2825)
        ]

        for start, destination in pairs:

            with self.subTest(
                start=start,
                destination=destination
            ):

                result = astar(
                    self.graph,
                    self.coordinates,
                    start,
                    destination
                )

                try:
                    expected = nx.shortest_path_length(
                        self.graph,
                        start,
                        destination,
                        weight="travel_time_seconds"
                    )

                except nx.NetworkXNoPath:
                    self.assertIsNone(result["path"])
                    self.assertIsNone(
                        result["travel_time_seconds"]
                    )

                else:
                    self.assertIsNotNone(result["path"])

                    self.assertIsNotNone(
                        result["travel_time_seconds"]
                    )

                    self.assertAlmostEqual(
                        result["travel_time_seconds"],
                        expected,
                        places=3
                    )

    def test_same_origin_destination(self):

        result = astar(
            self.graph,
            self.coordinates,
            2000,
            2000
        )

        self.assertEqual(result["path"], [2000])
        self.assertEqual(result["travel_time_seconds"], 0)

    def test_invalid_origin(self):

        result = astar(
            self.graph,
            self.coordinates,
            -1,
            3002
        )

        self.assertIsNone(result["path"])
        self.assertIsNone(result["travel_time_seconds"])

    def test_invalid_destination(self):

        result = astar(
            self.graph,
            self.coordinates,
            2000,
            -1
        )

        self.assertIsNone(result["path"])
        self.assertIsNone(result["travel_time_seconds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
