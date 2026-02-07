# Dr.Dropin Route Builder

Build shortest running routes that touch Dr.Dropin clinics in:
- Oslo (phase 1)
- Oslo + Sandvika (phase 2)

Current workspace includes:
- cleaned clinic datasets in `data/`
- verification maps in `art/`
- map generation script in `scripts/generate_verification_maps.py`
- interactive overlay map script in `scripts/generate_leaflet_maps.py`

## Tech Stack
- Python 3.11+
- UV for environment/dependency management
- OpenStreetMap-based routing stack (`osmnx`, `networkx`, `geopy`)
- GPX export (`gpxpy`)

## Quickstart (UV)
1. Install UV (if missing):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Sync environment:
```bash
uv sync --extra dev
```

3. Generate coordinate verification maps:
```bash
uv run python scripts/generate_verification_maps.py
```

Outputs:
- `art/map_oslo_verification.svg`
- `art/map_oslo_sandvika_verification.svg`

4. Generate interactive coordinate overlays on a real map (Leaflet):
```bash
uv run --no-project python scripts/generate_leaflet_maps.py
```

Outputs:
- `art/map_oslo_overlay.html`
- `art/map_oslo_sandvika_overlay.html`

## Data Files
- `data/drdropin_clinics_oslo_routing_ready.csv`
- `data/drdropin_clinics_oslo_sandvika_routing_ready.csv`
- `data/route_start_point.csv`

Schema and cleaning rules are documented in `docs/README.md`.

## Dependency Files
- `pyproject.toml` (primary source for UV)
- `requirements.txt` (runtime mirror)
- `requirements-dev.txt` (runtime + dev tooling)

## Notes
- Routing logic and GPX generation implementation are intentionally pending until approval.
- Verification maps are for coordinate placement checks only, not final road-network routes.
