.PHONY: install install-dev install-server install-benchmarks build build-gpu build-cpu run test benchmark docker-up docker-down clean

# Installation
install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev,server,tests,benchmarks,longform]"

install-server:
	uv pip install -e ".[server]"

install-benchmarks:
	uv pip install -e ".[benchmarks,tests]"

# Running
run:
	uvicorn gigaam_server.main:app --reload --host 0.0.0.0 --port 8000

run-cpu:
	GIGAAM_DEVICE=cpu GIGAAM_FP16_ENCODER=false uvicorn gigaam_server.main:app --reload

# Testing
test:
	pytest tests/ -v

benchmark:
	pytest gigaam_server/benchmarks/test_benchmark.py -v --benchmark-only

# Docker
build:
	docker-compose build

build-gpu:
	docker-compose build

build-cpu:
	docker build -f Dockerfile.cpu -t gigaam-cpu .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf build dist *.egg-info
