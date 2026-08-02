.PHONY: build up down bash logs test lint format format-check ci validate install demo-dry-run api smoke

build:
	docker compose build

up:
	docker compose up

down:
	docker compose down

bash:
	docker exec -it ew-notebook bash

logs:
	docker logs -f ew-notebook

test:
	python -m pytest -q

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

ci: lint format-check test

validate:
	python -m ew_housing_energy_impact validate

install:
	pip install -r requirements.txt
	pip install -e .

demo-dry-run:
	PYTHONPATH=src:. python -m ew_housing_energy_impact orchestrate --dry-run
	PYTHONPATH=src:. python -m ew_housing_energy_impact monitor --dry-run
	PYTHONPATH=src:. python -m ew_housing_energy_impact db-load --dry-run

api:
	uvicorn api:app --reload --port 8000

smoke:
	PYTHONPATH=src:. python -m ew_housing_energy_impact --help
	PYTHONPATH=src:. python -m ew_housing_energy_impact orchestrate --dry-run
	PYTHONPATH=src:. python -m ew_housing_energy_impact monitor --dry-run
	PYTHONPATH=src:. python -m ew_housing_energy_impact db-load --dry-run
