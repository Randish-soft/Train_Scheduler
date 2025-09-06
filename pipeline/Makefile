# pipeline/Makefile
# Simple automation for Train_Scheduler pipeline

PYTHON := python3
CLI := $(PYTHON) -m pipeline.cli

CONFIG ?= pipeline/config/belgium.example.yaml
SCHEMA ?= pipeline/config/schema/scenario.schema.json
MODELS_DIR ?= artifacts/models

.PHONY: all learn infer full lint clean

all: full

learn:
	$(CLI) learn --config $(CONFIG) --schema $(SCHEMA)

infer:
	$(CLI) infer --config $(CONFIG) --schema $(SCHEMA) --models-dir $(MODELS_DIR)

full:
	$(CLI) full --config $(CONFIG) --schema $(SCHEMA)

lint:
	ruff check ..
	black --check ..

clean:
	rm -rf artifacts/*
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
