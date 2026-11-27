
import unittest

from routing.route_service import get_routes


class TestRouteService(unittest.TestCase):

    def setUp(self):
        self.routes = get_routes(2000, 3002)

    def test_five_routes(self):
        self.assertEqual(len(self.routes), 5)

    def test_route_structure(self):
        required = {
            "route_number",
            "scats_sites",
            "distance_km",
            "travel_time_seconds",
            "status",
            "flow_sources"
        }

        for route in self.routes:
            self.assertEqual(set(route.keys()), required)

    def test_route_numbering(self):
        numbers = [
            route["route_number"]
            for route in self.routes
        ]

        self.assertEqual(numbers, [1, 2, 3, 4, 5])

    def test_distance_nonnegative(self):
        for route in self.routes:
            self.assertGreaterEqual(
                route["distance_km"], 0
            )

    def test_travel_time_positive(self):
        for route in self.routes:
            self.assertGreater(
                route["travel_time_seconds"], 0
            )

    def test_routes_sorted_by_travel_time(self):
        times = [
            route["travel_time_seconds"]
            for route in self.routes
        ]

        self.assertEqual(times, sorted(times))

    def test_traffic_status(self):
        for route in self.routes:
            self.assertEqual(
                route["status"],
                "Traffic-aware"
            )

    def test_flow_sources(self):
        for route in self.routes:
            self.assertIn(
                "Random Forest prediction",
                route["flow_sources"]
            )

    def test_invalid_start(self):
        routes = get_routes(999999, 3002)
        self.assertEqual(routes, [])

    def test_invalid_destination(self):
        routes = get_routes(2000, 999999)
        self.assertEqual(routes, [])

    def test_same_start_destination(self):
        routes = get_routes(2000, 2000)
        self.assertEqual(len(routes), 1)
        self.assertEqual(
            routes[0]["scats_sites"], [2000]
        )

    def test_distance_only_mode(self):
        routes = get_routes(
            2000,
            3002,
            use_traffic=False
        )

        self.assertEqual(len(routes), 5)

        for route in routes:
            self.assertIsNone(
                route["travel_time_seconds"]
            )

            self.assertEqual(
                route["status"],
                "Distance-only prototype"
            )

            self.assertNotIn(
                "flow_sources",
                route
            )


    def test_historical_fallback_restores_routes(self):
        routes = get_routes(970, 2200)

        self.assertGreater(len(routes), 0)

        self.assertIn(
            "Observed historical fallback",
            routes[0]["flow_sources"]
        )

    def test_missing_prediction_sites_supported(self):
        for start, end in [
            (3685, 2000),
            (4264, 3002)
        ]:
            routes = get_routes(start, end)

            self.assertGreater(len(routes), 0)

            self.assertIn(
                "Observed historical fallback",
                routes[0]["flow_sources"]
            )

if __name__ == "__main__":
    unittest.main()
