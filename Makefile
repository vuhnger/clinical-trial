.PHONY: sync maps overlays routes

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
