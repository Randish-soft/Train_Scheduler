# Using Model (AI based)
## Generate:
```bash
poetry run python -m Model --mode generate --input input/lebanon_cities.csv --country lebanon --optimize "cost,ridership" --output-dir output/lebanon
```

## Learn:
```bash
python -m Model --mode learn --country Country_Name --train-types IC,OTHERS
```
### Example:
```bash
python -m Model --mode learn --country DE --train-types IC,ICE,R,S,U
```

### With Debug -- Resource Hogger!!!
```bash
python -m Model --mode learn --country DE --train-types IC --log-level debug --verbose
```

### Belgium Learn (Stable):
```bash
python -m Model --mode learn --country belgium --train-types IC --log-level debug --verbose
```

### Helper:
```bash
python -m Model --help 
```

# Using the pipeline (raster)
python pipeline/__main__.py

## OR
python pipeline/pipeline_initiator.py

## Example Usage:
Full Scope:
python pipeline/pipeline_initiator.py --country "Italy" \
  --budget 1000000000 \
  --min-cities 5 \
  --max-cities 20 \
  --urban-focus \
  --generate-reports \
  --output-dir ./my_output

## Dry Run
python pipeline/pipeline_initiator.py --country "Netherlands" --dry-run --verbose

## Testing 
python pipeline/pipeline_initiator.py --country "Netherlands" --generate-reports --verbose