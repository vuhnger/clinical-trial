# Product Requirements Document (PRD)

## Product Name
Dr.Dropin Route Builder (Oslo + Sandvika)

## Status
Draft v1.0

## Goal
Build a simple Python tool that:
1. Finds all Dr.Dropin clinic locations in Oslo and Sandvika.
2. Converts clinic addresses into geographic coordinates (latitude/longitude).
3. Creates two GPX route files:
   - Route A: touches every clinic in Oslo.
   - Route B: touches every clinic in Oslo and Sandvika.
4. Optimizes each route for shortest total travel distance on the street/path network.
5. Uses a fixed start point for both routes: `Sørkedalsveien 8a, 0369 Oslo`.
6. Produces a clean route visualization as PNG or SVG.

## Background
The user needs route files for Strava that pass all relevant Dr.Dropin clinics. Data collection and route generation should be automated where possible, with a manual fallback if some clinic coordinates cannot be resolved.

## Scope
### In Scope
- Programmatic collection of clinic data (name, city, address).
- Geocoding addresses to coordinates.
- Deduplication/validation of clinic locations.
- Route generation using road/path network data.
- GPX export compatible with Strava import.
- CLI execution with clear output and logs.
- Route optimization focused on shortest route that visits each clinic location once.
- Static map export (PNG and/or SVG) for route presentation.

### Out of Scope
- UI/web frontend.
- Real-time traffic optimization.
- Turn-by-turn navigation instructions.
- Ongoing production hosting.
- Third-party clinic data sources (only official Dr.Dropin sources are used).

## Users
- Primary user: project owner running scripts locally.

## Functional Requirements
1. Clinic Discovery
   - The tool must gather Dr.Dropin clinic entries for Oslo and Sandvika.
   - Source policy: fetch only from official Dr.Dropin site(s) (machine-readable source, API, or structured HTML).
   - Fallback: allow manual seed file (`data/clinics_manual.csv`) with clinic name/address/city.
   - Geographic inclusion should use municipality boundary filtering when possible.

2. Address Normalization
   - The tool must normalize address strings into a consistent format suitable for geocoding.
   - The tool should append country context (`Norway`) when missing.
   - Duplicate clinic entries that share the same physical address/coordinate must be cleaned at dataset level before routing.

3. Geocoding
   - The tool must attempt geocoding for every clinic address.
   - Suggested provider: OpenStreetMap Nominatim (or equivalent OSM geocoder).
   - If geocoding fails, the tool must mark unresolved entries in an output report.
   - The tool must support manual coordinate overrides from file.

4. Route Construction
   - Fixed route start node for both routes: `Sørkedalsveien 8a, 0369 Oslo`.
   - Route form: open route (fixed start, flexible end) to minimize total distance.
   - Route A: include all resolved Oslo clinics as route waypoints.
   - Route B: include all resolved Oslo + Sandvika clinics as route waypoints.
   - Route must follow the pedestrian/running street-path network (no cycling profile, no straight-line segments).
   - Pavement preference: prefer paved walkable edges where OSM `surface`/related tags are available.
   - Route objective must minimize total network distance.
   - Each clinic destination must be visited only once (strict no-dupe policy).
   - Suggested implementation: OSMnx + NetworkX + TSP approximation/solver.

5. GPX Export
   - Produce two files:
     - `output/oslo_clinics_route.gpx`
     - `output/oslo_sandvika_clinics_route.gpx`
   - Files must import into Strava as routes.

6. Map Visualization Export
   - Export at least one clean map per route as static image:
     - `output/oslo_clinics_route.png` (or `.svg`)
     - `output/oslo_sandvika_clinics_route.png` (or `.svg`)
   - Map should clearly show route line, clinic stops, and start location.

7. Logging & Reporting
   - Print summary:
     - Clinics found
     - Clinics geocoded
     - Clinics unresolved
     - Output file paths
   - Save unresolved entries to `output/unresolved_clinics.csv`.

## Non-Functional Requirements
- Language: Python 3.11+.
- Runtime target: under 5 minutes for normal network conditions.
- Reproducibility: deterministic outputs for same input data and configuration.
- Maintainability: modular code (`fetch`, `geocode`, `route`, `export`).
- Compliance: respect geocoding provider usage limits and attribution requirements.

## Proposed Tech Stack
- Python
- `requests` and `beautifulsoup4` (clinic data extraction)
- `pandas` (data handling)
- `geopy` (Nominatim geocoding wrapper) or direct Nominatim calls
- `osmnx` + `networkx` (road network + routing)
- `gpxpy` (GPX writing)
- `matplotlib` (+ optional `contextily`) for clean static map output (PNG/SVG)
- `typer` or `argparse` (CLI)

## Data Model
Each clinic record should include:
- `clinic_name`
- `city` (`oslo` or `sandvika`)
- `address`
- `postcode` (optional)
- `country` (`Norway`)
- `latitude`
- `longitude`
- `source` (scraped/manual)
- `status` (`resolved`/`unresolved`)

## High-Level Workflow
1. Collect clinic list (auto + optional manual supplement).
2. Normalize and deduplicate addresses.
3. Geocode addresses.
4. Build network graph covering Oslo/Sandvika area.
5. Snap clinics to nearest routable nodes.
6. Compute route path touching all required clinics.
7. Export GPX files and unresolved report.

## Routing Strategy (MVP)
- Build an OSM street/path graph for the relevant region.
- Snap each unique clinic coordinate to nearest routable node.
- Compute pairwise shortest-path distances between clinic nodes.
- Solve visit order with a shortest-route algorithm:
  - Preferred: TSP-style optimization (e.g., Christofides approximation via NetworkX, which uses MST internally).
  - Optional for small N: exact solver via OR-Tools.
- Expand optimized visit order back into full node-by-node network paths.
- Merge path segments into one continuous GPX track.
- If disconnected segments occur, log and skip with explicit warning.

## Delivery Phases
1. Phase 1 (Oslo only)
   - Complete end-to-end data fetch, geocode, route optimization, GPX export, and map export for Oslo.
2. Phase 2 (Oslo + Sandvika)
   - Extend validated Oslo pipeline to include Sandvika clinics in combined route output.

## CLI Requirements
Example command:
```bash
python -m app.main build-routes --country "Norway"
```

Suggested options:
- `--use-manual-overrides`
- `--max-geocode-retries`
- `--output-dir`
- `--city-filter oslo|sandvika|both`

## Success Metrics
- 100% of known clinics included in the generated city-specific route when coordinates are available.
- 0 duplicate clinic visits in output route sequence.
- GPX files import into Strava without schema errors.
- Map exports are legible and presentation-ready for company event communication.
- Unresolved clinics are clearly listed for manual correction.

## Risks and Mitigations
1. Source website structure changes
   - Mitigation: isolate parser logic and add manual CSV fallback.

2. Geocoder rate limits / failed lookups
   - Mitigation: retry with backoff, cache results, manual override file.

3. Incomplete/ambiguous addresses
   - Mitigation: append city/country context and support manual correction.

4. Route graph gaps or disconnected network
   - Mitigation: detect gaps, log warnings, export partial route plus exception report.

## Deliverables
1. `PRD.md`
2. Python project structure with runnable CLI.
3. Generated data artifacts:
   - `output/oslo_clinics_route.gpx`
   - `output/oslo_sandvika_clinics_route.gpx`
   - `output/oslo_clinics_route.png` (or `.svg`)
   - `output/oslo_sandvika_clinics_route.png` (or `.svg`)
   - `output/unresolved_clinics.csv` (if needed)

## Acceptance Criteria
1. Running the tool creates both GPX files in `output/`.
2. Oslo route includes all geocoded Oslo clinics.
3. Oslo+Sandvika route includes all geocoded clinics from both areas.
4. Each route visits every included clinic building/coordinate exactly once (no duplicates).
5. Route order is produced by a shortest-route optimization algorithm (not arbitrary ordering).
6. Both routes start at `Sørkedalsveien 8a, 0369 Oslo`.
7. Running/pedestrian routing profile is used (not cycling).
8. A clean static map export (PNG or SVG) is generated for each route.
9. Any clinic that fails geocoding appears in `output/unresolved_clinics.csv`.
10. At least one documented mechanism exists for manual coordinate override.

## Clarification
Yes, your description is clear enough to create this PRD and start implementation.
