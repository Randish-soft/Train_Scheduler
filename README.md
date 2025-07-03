BCPC — Bring Cities Back to the People (terrain‑aware rail planner)

1  Quick Start (local)

cd Train_Scheduler
poetry install --no-interaction
poetry run python -m src --csv input/lebanon_cities_2024.csv

First run downloads SRTM DEM tiles to data/dem/, fetches OSM boundaries and writes results to output/.

2  Folder layout

input/      scenario CSVs
output/     GeoJSON or *_EMPTY.txt per city
data/       cache → dem/  _boundaries/
notebooks/  demos
src/        all modules
tests/      pytest stubs

3  Scenario CSV columns

column

unit

example

city_id

str

LB-BEY

city_name

str

Beirut

population

int

1 916 100

tourism_index

0‑1

0.7

daily_commuters

int

574 830

budget_total_eur

€

2 000 000 000

Optional: station_origin_lon/lat, station_dest_lon/lat to force termini.

4  CLI usage

--csv <file>   run that scenario file
(no flag)      arrow‑select when multiple CSVs present

5  Docker / DevContainer

docker compose up --build   # Dagster web at localhost:3000

Volumes map output/ back to host.

6  Routing internals

OSM polygon (or 10 km buffered point).

DEM (AWS SRTM1) → slope grid.

Overpass rail filter "railway"~"rail|light_rail|subway|tram" (fallback road).

A* with cost = length × slope‑penalty.

Curvature vs TrackType.min_radius_m.

7  Customising catalogues

Edit src/models.py + src/optimise.py:

TrackType("Std‑Cat‑200", Gauge.STANDARD, True, 200, 1800, 14_000_000)
TrainType("6‑car EMU",  Gauge.STANDARD, 600, 200, 15_000_000, 9)

8  Outputs

file

meaning

*_tracks.geojson

real coordinates, property length_km

*_EMPTY.txt

budget too low, no network

*_ERROR.txt

export or routing failure

9  Troubleshooting

symptom

fix

404 on SRTM tile

some 1° cells missing → switch to SRTM3 or ignore; flat fallback will run

Overpass busy

re‑run or set ox.settings.overpass_endpoint to a mirror

"Graph has no edges"

city has no mapped rails – road fallback already engaged

10  Maintenance

poetry update                # upgrade deps
rm -r data/dem/*             # purge DEM cache
rm -r data/_boundaries/*     # purge OSM cache

© 2025 BCPC dev team — MIT licence

