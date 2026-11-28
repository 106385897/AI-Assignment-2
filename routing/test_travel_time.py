import unittest

from routing.travel_time import (
    to_hourly_flow,
    flow_to_speed,
    calculate_travel_time,
    calculate_edge_cost
)


class TestTravelTime(unittest.TestCase):

    def test_hourly_flow(self):
        self.assertEqual(to_hourly_flow(100), 400)

    def test_zero_flow(self):
        self.assertEqual(flow_to_speed(0), 60)

    def test_speed_limit(self):
        self.assertLessEqual(flow_to_speed(400), 60)

    def test_road_capacity(self):
        self.assertAlmostEqual(flow_to_speed(1500), 32, delta=0.1)

    def test_intersection_delay(self):
        self.assertAlmostEqual(
            calculate_travel_time(1, 0), 90
        )

    def test_higher_traffic(self):
        low = calculate_edge_cost(1, 100)
        high = calculate_edge_cost(1, 250)

        self.assertGreater(high, low)

    def test_edge_cost(self):
        result = calculate_edge_cost(1, 100)

        self.assertGreater(result, 30)

    def test_negative_flow(self):
        with self.assertRaises(ValueError):
            to_hourly_flow(-10)

    def test_over_capacity(self):
        with self.assertRaises(ValueError):
            flow_to_speed(1600)

    def test_negative_distance(self):
        with self.assertRaises(ValueError):
            calculate_travel_time(-1, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)