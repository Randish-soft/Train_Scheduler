# Acknowledge and prepare the refactored plan for intercity_planner.py
plan = """
Refactor Plan:

1. Use city lat/lon/elevation directly from the input CSV — no hardcoded coordinates.
2. Use OpenTopography DEM with your API key to extract elevation and terrain.
3. Use `trace_route` with DEM cost surface routing to trace realistic rail paths (avoiding mountains, rivers, etc.).
4. Estimate elevation changes and tunnel requirements based on the traced geometry, not straight lines.
5. Final output GeoDataFrame includes connection type (above-ground, on-ground, tunnel) in feature properties.

Steps:
- Extend `load_scenario()` to return `lat`, `lon`, `elevation`.
- Replace geodesic line with DEM-aware `trace_route` in `generate_all_possible_connections()`.
- Compute elevation change from DEM profiles along the traced path.
- Use OpenTopography API to fetch DEM for bounding box of Lebanon or route corridors.
- Store API key securely from `.env` using `os.getenv("KEY")`.

Implementation follows next.
"""

