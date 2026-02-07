# Dataset Documentation

## Overview
This project contains clinic datasets for building shortest running routes that:
- start at `Sørkedalsveien 8a, 0369 Oslo`
- visit each clinic coordinate once (no duplicate coordinates)
- support phase 1 (`Oslo`) and phase 2 (`Oslo + Sandvika`)

Source data was extracted from Dr.Dropin official clinic data:
- `https://www.drdropin.no/klinikker`
- `https://www.drdropin.no/page-data/sq/d/2676090211.json`

## Files
- `data/drdropin_clinics_oslo_sandvika_raw.csv`
  - Raw extracted clinic list for Oslo + Sandvika scope.
- `data/drdropin_clinics_oslo_sandvika_qc.csv`
  - Coordinate quality checks.
- `data/drdropin_clinics_oslo_routing_ready.csv`
  - Cleaned routing input for Oslo-only route (phase 1).
- `data/drdropin_clinics_oslo_sandvika_routing_ready.csv`
  - Cleaned routing input for Oslo + Sandvika route (phase 2).
- `data/route_start_point.csv`
  - Fixed route start point dataset.

## Routing-Ready Schema
`drdropin_clinics_oslo_routing_ready.csv` and `drdropin_clinics_oslo_sandvika_routing_ready.csv` contain:
- `clinic_name`
- `street`
- `postal_code`
- `municipality`
- `area` (`Oslo` or `Sandvika`)
- `country`
- `latitude`
- `longitude`
- `source`
- `source_dataset`
- `coordinate_source`
- `coordinate_override_reason`
- `include_in_routing`

## Data Cleaning Rules
- Scope filter: only `Oslo` and `Sandvika` (Sandvika represented in source municipality `Bærum`).
- Deduplication policy: unique coordinates only for routing inputs.
- Start-point correction:
  - `Bedriftshelsetjeneste` had an official coordinate that appears misplaced.
  - Replaced with OSM Nominatim result for `Sørkedalsveien 8A, 0369 Oslo`:
    - `lat=59.9304610`
    - `lon=10.7115718`

## How To Use
1. Use `data/route_start_point.csv` as the route start node.
2. For phase 1, load `data/drdropin_clinics_oslo_routing_ready.csv`.
3. For phase 2, load `data/drdropin_clinics_oslo_sandvika_routing_ready.csv`.
4. Build shortest-path ordering on pedestrian/running network.
5. Export GPX and map outputs.

## Notes
- All coordinates are WGS84 decimal degrees.
- `include_in_routing` is currently `yes` for all rows in routing-ready files.
