.PHONY: sync maps overlays routes backend-dev docker-up docker-down

sync:
	uv sync --extra dev

maps:
	uv run python scripts/generate_verification_maps.py
	uv run --no-project python scripts/generate_leaflet_maps.py

overlays:
	uv run --no-project python scripts/build_routes.py
	uv run --no-project python scripts/generate_leaflet_maps.py

routes:
	uv run --no-project python scripts/build_routes.py
	uv run --no-project python scripts/generate_leaflet_maps.py

backend-dev:
	UV_CACHE_DIR=.uv-cache uv run --no-project uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up --build -d
	@echo "Frontend: http://localhost:8080"
	@echo "Backend:  http://localhost:8000 (health: /api/health)"

docker-down:
	docker compose down
