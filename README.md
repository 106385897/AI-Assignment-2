# AI Assignment 2

COS30019 - Introduction to Artificial Intelligence

## Routing System

The routing system uses SCATS intersection locations and OpenStreetMap road data to generate alternative routes through Boroondara, Melbourne.

The current implementation supports distance-based routing. Traffic-based travel time calculations are still being integrated.

### Main Files

- `routing/top_five_routes.py` - Finds up to five routes between two SCATS intersections.
- `routing/route_service.py` - Provides route results for the GUI.
- `routing/travel_time.py` - Converts traffic flow into estimated travel time.
- `routing/build_road_network.py` - Downloads and saves the OpenStreetMap road network.
- `routing/generate_scats_graph.py` - Generates candidate SCATS connections.
- `routing/filter_scats_graph.py` - Filters connections identified by the route audit.
- `data/scats_filtered_connections.csv` - Stores the current provisional SCATS graph.

### Running the Routing System

Run commands from the main project folder.

To display five routes:

```bash
python routing/top_five_routes.py
```

To test the GUI routing interface:

```bash
python -m routing.route_service
```

### Using the Routing Interface

Other modules can import the routing service:

```python
from routing.route_service import get_routes

routes = get_routes(2000, 3002)

for route in routes:
    print(route)
```

Each result contains:

- `route_number` - Route ranking.
- `scats_sites` - Ordered SCATS intersections.
- `distance_km` - Total estimated distance.
- `travel_time_seconds` - Estimated travel time, currently unavailable.
- `status` - Current routing mode.

Invalid SCATS numbers return an empty list.

If the starting point and destination are the same, the service returns a zero-distance route.

### Testing

Run the routing tests:

```bash
python routing/test_top_five_routes.py
python routing/test_travel_time.py
python routing/test_route_traffic.py
```

The current implementation has passed 26 tests across these three modules.

### Current Limitations

- Routes are currently ranked by distance, not predicted travel time.
- The SCATS connection graph is provisional and requires further road validation.
- Traffic predictions are aggregated across multiple intersection detectors.
- Some predicted flows exceed the conversion formula's assumed road capacity.
- SCATS 3685 and 4264 have no predictions in the shared test dataset.
- The correct traffic-flow conversion method is awaiting tutor clarification.
- The full road-network generation process uses additional source data and a locally saved OpenStreetMap graph.

These limitations will be addressed during the final integration and validation stages.
