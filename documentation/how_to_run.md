
# Generate:
```bash
poetry run python -m Model --mode generate --input input/lebanon_cities.csv --country lebanon --optimize "cost,ridership" --output-dir output/lebanon
```

# Learn:
```bash
python -m Model --mode learn --country Country_Name --train-types IC,OTHERS
```
## Example:
```bash
python -m Model --mode learn --country DE --train-types IC,ICE,R,S,U
```

## With Debug -- Resource Hogger!!!
```bash
python -m Model --mode learn --country DE --train-types IC --log-level debug --verbose
```

## Belgium Learn (Stable):
```bash
python -m Model --mode learn --country belgium --train-types IC --log-level debug --verbose
```

## Helper:
```bash
python -m Model --help 
```