
# Generate:

poetry run python -m Model --mode generate --input input/lebanon_cities.csv --country lebanon --optimize "cost,ridership" --output-dir output/lebanon

# Learn:
python -m Model --mode learn --country Country_Name --train-types IC,OTHERS

## Example:
python -m Model --mode learn --country DE --train-types IC,ICE,R,S,U

## With Debug:
python -m Model --mode learn --country DE --train-types IC --log-level debug --verbose

## Belgium Learn (Stable):
python -m Model --mode learn --country belgium --train-types IC --log-level debug --verbose