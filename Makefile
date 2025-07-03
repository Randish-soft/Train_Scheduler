.PHONY: help build up down run clean

help:
	@echo "Commands:"
	@echo "  make build  - Build Docker images"
	@echo "  make up     - Start services"
	@echo "  make down   - Stop services"
	@echo "  make run    - Run pipeline"
	@echo "  make clean  - Clean outputs"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

run:
	docker-compose run --rm bcpc-app python -m src.bcpc_pipeline

clean:
	rm -rf outputs/* logs/* __pycache__
