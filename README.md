# Dr.Dropin Route Builder

Builds shortest practical running loops that:
- start at `Sørkedalsveien 8a, 0369 Oslo`
- visit all Dr.Dropin clinic points once (Oslo or Oslo+Sandvika scope)
- return to the start point
- export GPX + HTML map overlays

## Project Layout
- `data/`: cleaned routing datasets
- `scripts/`: route and map generation scripts
- `output/`: generated GPX, route paths, summaries
- `art/`: generated map visuals and logo asset
- `docs/README.md`: dataset schema/cleaning details

## Prerequisites
- Python 3.11+
- `uv`
- Internet access (required for OpenStreetMap/Overpass route graph download)

Install `uv` if needed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup
```bash
make sync
```
This runs:
```bash
uv sync --extra dev
```

## Run Everything
Primary command:
```bash
make routes
```

This will:
1. Build road-following walking loops from OSM network.
2. Export GPX files.
3. Regenerate HTML overlay maps (same filenames, replaced in place).

## Command Reference
1. Sync environment:
```bash
make sync
```

2. Build routes + regenerate overlays:
```bash
make routes
```

3. Generate both verification SVG maps and HTML overlays:
```bash
make maps
```

4. Alias for route build + overlays:
```bash
make overlays
```

## Outputs
### Routes
- `output/oslo_clinics_route.gpx`
- `output/oslo_sandvika_clinics_route.gpx`
- `output/oslo_route_order.csv`
- `output/oslo_sandvika_route_order.csv`
- `output/oslo_route_path.json`
- `output/oslo_sandvika_route_path.json`
- `output/route_summary.json`

### Maps
- `art/map_oslo_overlay.html`
- `art/map_oslo_sandvika_overlay.html`
- `art/map_oslo_verification.svg`
- `art/map_oslo_sandvika_verification.svg`

## Optimization Controls
You can run a longer optimization search by setting environment variables:
```bash
ROUTE_RANDOM_STARTS=15000 ROUTE_TWO_OPT_ROUNDS=900 \
UV_CACHE_DIR=.uv-cache MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache \
uv run --no-project python scripts/build_routes.py
```

Larger values generally improve routes but increase runtime.

## Data Inputs
- `data/drdropin_clinics_oslo_routing_ready.csv`
- `data/drdropin_clinics_oslo_sandvika_routing_ready.csv`
- `data/route_start_point.csv`

Dataset structure and cleaning policy are documented in `docs/README.md`.
