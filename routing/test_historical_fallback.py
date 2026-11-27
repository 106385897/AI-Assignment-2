
import unittest

from routing.top_five_routes import load_graph, find_top_five_routes


class TestHistoricalFallback(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.normal_graph = load_graph(
            use_traffic=True
        )

        cls.fallback_graph = load_graph(
            use_traffic=True,
            allow_historical_fallback=True
        )

    def test_fallback_adds_connections(self):

        self.assertGreater(
            self.fallback_graph.number_of_edges(),
            self.normal_graph.number_of_edges()
        )

    def test_fallback_restores_3685(self):

        edges = list(
            self.fallback_graph.out_edges(
                3685,
                data=True
            )
        )

        self.assertGreater(len(edges), 0)

        for _, _, data in edges:
            self.assertEqual(
                data["flow_source"],
                "Observed historical fallback"
            )

    def test_fallback_restores_4264(self):

        edges = list(
            self.fallback_graph.out_edges(
                4264,
                data=True
            )
        )

        self.assertGreater(len(edges), 0)

        for _, _, data in edges:
            self.assertEqual(
                data["flow_source"],
                "Observed historical fallback"
            )

    def test_fallback_disabled_by_default(self):

        self.assertEqual(
            self.normal_graph.out_degree(3685),
            0
        )

        self.assertEqual(
            self.normal_graph.out_degree(4264),
            0
        )

    def test_fallback_routes_labelled(self):

        routes = find_top_five_routes(
            self.fallback_graph,
            4264,
            3002,
            use_traffic=True
        )

        self.assertGreater(len(routes), 0)

        self.assertIn(
            "Observed historical fallback",
            routes[0]["flow_sources"]
        )

    def test_existing_predictions_preserved(self):

        edges = list(
            self.fallback_graph.out_edges(
                2000,
                data=True
            )
        )

        self.assertGreater(len(edges), 0)

        for _, _, data in edges:
            self.assertEqual(
                data["flow_source"],
                "Random Forest prediction"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
