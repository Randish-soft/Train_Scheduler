# Pipeline Usage

This pipeline can now fetch required datasets **automatically from online sources**  
(OSM railways via Overpass + Nominatim, and SRTM topography via the `elevation` package).  
You can still point to local files in your config if you prefer.

---

## Run with Makefile

```bash
# Run training (learn stage: ingest + features + train)
make learn CONFIG=pipeline/config/belgium.example.yaml

# Run inference (ingest + features + route + timetable + report)
make infer CONFIG=pipeline/config/belgium.example.yaml MODELS_DIR=artifacts/models

# Run full pipeline (learn + infer)
make full CONFIG=pipeline/config/belgium.example.yaml

# Run with CLI

# Learn stage only
python -m pipeline.cli learn --config pipeline/config/belgium.example.yaml --schema pipeline/config/schema/scenario.schema.json


# No Schema:
python -m pipeline.cli learn --config pipeline/config/belgium.example.yaml

# Inference stage only
python -m pipeline.cli infer --config pipeline/config/belgium.example.yaml --models-dir artifacts/models

# Full pipeline (learn + infer)
python -m pipeline.cli full --config pipeline/config/belgium.example.yaml


# Cleaning
make clean

#Pipeline (new)
python run.py