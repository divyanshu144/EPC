.PHONY: build up down bash logs test validate install

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
	pytest

validate:
	python -m ew_housing_energy_impact validate

install:
	pip install -r requirements.txt
	pip install -e .
