# BCPC — Bring Cities back to the People (not Cars)

BCPC is a modular, data-driven toolkit that designs people-centred rail networks:

* **CSV ➜ Maps** Ingest simple scenario CSVs, enrich with open data, run
  multi-objective optimisation, then export interactive OpenStreetMap layers.
* **Budget-aware** Costs calibrated from real-world tenders and indices.
* **Multi-modal** Adds tram, bus, and active-travel links around core rail.
* **Headless or interactive** Run as a CLI / FastAPI micro-service or explore in notebooks.

---

## Quick start (with Docker + Poetry)

```bash
docker compose up --build
```
# Project layout
data/       raw & interim geo/CSV inputs
input/      user-supplied scenario CSVs
output/     generated maps, reports, BoMs
src/        application code
tests/      pytest suite


