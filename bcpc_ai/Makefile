.PHONY: help setup train evaluate predict serve clean test lint format

help:
	@echo "Available commands:"
	@echo "  setup      - Install dependencies and setup environment"
	@echo "  train      - Train the model"
	@echo "  evaluate   - Evaluate model performance"
	@echo "  predict    - Generate predictions for a country"
	@echo "  serve      - Start API server"
	@echo "  clean      - Clean generated files and caches"
	@echo "  test       - Run unit tests"
	@echo "  lint       - Run code linting"
	@echo "  format     - Format code with black"

setup:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	mkdir -p data/train data/test models/checkpoints artifacts/figures artifacts/reports logs

train:
	python train.py \
		--config configs/training/schedule.yaml \
		--model-config configs/model/architecture.yaml \
		--epochs 100 \
		--batch-size 32

train-quick:
	python train.py \
		--epochs 10 \
		--batch-size 64

evaluate:
	python evaluate.py \
		--model models/checkpoints/best_model.pth \
		--config configs/model/architecture.yaml \
		--visualize

predict-lebanon:
	python predict.py \
		--country lebanon \
		--models-dir models/final \
		--report

predict-egypt:
	python predict.py \
		--country egypt \
		--models-dir models/final \
		--report

serve:
	python serve.py \
		--model models/final/best_model.pth \
		--config configs/model/architecture.yaml \
		--port 8000

serve-dev:
	python serve.py \
		--model models/checkpoints/best_model.pth \
		--config configs/model/architecture.yaml \
		--port 8000 \
		--debug

test:
	pytest tests/ -v --cov=src --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	flake8 src/ --max-line-length=100 --ignore=E203,W503
	pylint src/

format:
	black src/ tests/ *.py --line-length=100
	isort src/ tests/ *.py

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf *.egg-info build dist
	rm -rf logs/* artifacts/figures/* artifacts/reports/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

docker-build:
	docker build -t bcpc-railway:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models bcpc-railway:latest

data-download:
	@echo "Downloading sample data..."
	python scripts/download_data.py

data-prepare:
	@echo "Preparing data for training..."
	python scripts/prepare_data.py

benchmark:
	python scripts/benchmark.py --model models/final/best_model.pth

deploy:
	@echo "Deploying to production..."
	./scripts/deploy.sh

monitor:
	@echo "Starting monitoring dashboard..."
	python scripts/monitor.py

all: setup data-download data-prepare train evaluate