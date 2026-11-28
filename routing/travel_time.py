import math


MAX_SPEED = 60
MAX_FLOW = 1500
CAPACITY_SPEED = 32
INTERSECTION_DELAY = 30

A = -1.4648375
B = 93.75


def to_hourly_flow(flow_15min):
    if not math.isfinite(flow_15min) or flow_15min < 0:
        raise ValueError("Traffic flow must be a valid non-negative number")

    return flow_15min * 4


def flow_to_speed(hourly_flow):
    if not math.isfinite(hourly_flow) or hourly_flow < 0:
        raise ValueError("Traffic flow must be a valid non-negative number")

    if hourly_flow > MAX_FLOW:
        raise ValueError("Traffic flow exceeds the assumed road capacity")

    discriminant = B ** 2 + 4 * A * hourly_flow

    if discriminant < 0:
        raise ValueError("Cannot calculate speed for this traffic flow")

    speed = (-B - math.sqrt(discriminant)) / (2 * A)

    return min(speed, MAX_SPEED)


def calculate_travel_time(distance_km, hourly_flow):
    if not math.isfinite(distance_km) or distance_km < 0:
        raise ValueError("Distance must be a valid non-negative number")

    speed = flow_to_speed(hourly_flow)

    driving_time = (distance_km / speed) * 3600

    return driving_time + INTERSECTION_DELAY


def calculate_edge_cost(distance_km, predicted_flow_15min):
    hourly_flow = to_hourly_flow(predicted_flow_15min)

    return calculate_travel_time(distance_km, hourly_flow)