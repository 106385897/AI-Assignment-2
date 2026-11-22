
import unittest

from routing.route_service import get_routes


class TestRouteService(unittest.TestCase):

    def test_five_routes(self):
        routes = get_routes(2000, 3002)
        self.assertEqual(len(routes), 5)

    def test_route_structure(self):
        routes = get_routes(2000, 3002)

        required = {
            "route_number",
            "scats_sites",
            "distance_km",
            "travel_time_seconds",
            "status"
        }

        for route in routes:
            self.assertEqual(set(route.keys()), required)

    def test_invalid_start(self):
        self.assertEqual(get_routes(9999, 3002), [])

    def test_invalid_destination(self):
        self.assertEqual(get_routes(2000, 9999), [])

    def test_same_start_destination(self):
        routes = get_routes(2000, 2000)

        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["scats_sites"], [2000])
        self.assertEqual(routes[0]["distance_km"], 0)

    def test_distance_nonnegative(self):
        routes = get_routes(2000, 3002)

        for route in routes:
            self.assertGreaterEqual(route["distance_km"], 0)

    def test_travel_time_pending(self):
        routes = get_routes(2000, 3002)

        for route in routes:
            self.assertIsNone(route["travel_time_seconds"])
            self.assertEqual(
                route["status"],
                "Distance-only prototype"
            )

    def test_route_numbering(self):
        routes = get_routes(2000, 3002)

        numbers = [
            route["route_number"]
            for route in routes
        ]

        self.assertEqual(numbers, [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
