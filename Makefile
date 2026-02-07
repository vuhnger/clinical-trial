.PHONY: sync maps overlays

sync:
	uv sync --extra dev

maps:
	uv run python scripts/generate_verification_maps.py

overlays:
	uv run --no-project python scripts/generate_leaflet_maps.py
