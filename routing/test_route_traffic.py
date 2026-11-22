
import unittest
import pandas as pd

from top_five_routes import load_graph, find_top_five_routes


class TestRouteTraffic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.graph = load_graph()

        flows = pd.read_csv(
            "data/scats_average_predicted_flows.csv"
        )

        cls.flow_lookup = dict(
            zip(
                flows["SCATS"].astype(int),
                flows["PredictedFlow15Min"]
            )
        )

    def test_predictions_available(self):
        self.assertGreater(
            len(self.flow_lookup),
            0
        )

    def test_missing_site_4264(self):
        self.assertNotIn(
            4264,
            self.flow_lookup
        )

    def test_missing_site_3685(self):
        self.assertNotIn(
            3685,
            self.flow_lookup
        )

    def test_route_one_has_predictions(self):

        routes = find_top_five_routes(
            self.graph, 2000, 3002
        )

        missing = [
            site for site in routes[0]["route"]
            if site not in self.flow_lookup
        ]

        self.assertEqual(missing, [])

    def test_over_capacity_detected(self):

        over_capacity = [
            site for site, flow in self.flow_lookup.items()
            if flow * 4 > 1500
        ]

        self.assertGreater(
            len(over_capacity),
            0
        )

    def test_predicted_flows_nonnegative(self):

        for flow in self.flow_lookup.values():
            self.assertGreaterEqual(
                flow,
                0
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
