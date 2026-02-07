#!/usr/bin/env python3
"""Build traced loop routes and export GPX artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from route_tracing import (
    build_walk_graph,
    optimize_closed_route,
    route_length_m,
    route_on_walk_graph,
    write_gpx,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"

OSLO_CSV = DATA_DIR / "drdropin_clinics_oslo_routing_ready.csv"
OSLO_SANDVIKA_CSV = DATA_DIR / "drdropin_clinics_oslo_sandvika_routing_ready.csv"
START_CSV = DATA_DIR / "route_start_point.csv"

OSLO_GPX = OUT_DIR / "oslo_clinics_route.gpx"
OSLO_SANDVIKA_GPX = OUT_DIR / "oslo_sandvika_clinics_route.gpx"
SUMMARY_JSON = OUT_DIR / "route_summary.json"
OSLO_ORDER_CSV = OUT_DIR / "oslo_route_order.csv"
OSLO_SANDVIKA_ORDER_CSV = OUT_DIR / "oslo_sandvika_route_order.csv"
OSLO_PATH_JSON = OUT_DIR / "oslo_route_path.json"
OSLO_SANDVIKA_PATH_JSON = OUT_DIR / "oslo_sandvika_route_path.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_latlon(row: dict[str, str]) -> tuple[float, float]:
    return float(row["latitude"]), float(row["longitude"])


def write_route_order_csv(
    path: Path,
    route: list[tuple[float, float]],
    start_row: dict[str, str],
    clinics: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (lat, lon) in enumerate(route, start=1):
        name = "Main office start/end"
        street = start_row["street"]
        postal = start_row["postal_code"]
        municipality = start_row["municipality"]

        for clinic in clinics:
            clat, clon = to_latlon(clinic)
            if abs(clat - lat) <= 1e-7 and abs(clon - lon) <= 1e-7:
                name = clinic["clinic_name"]
                street = clinic["street"]
                postal = clinic["postal_code"]
                municipality = clinic["municipality"]
                break

        rows.append(
            {
                "sequence": idx,
                "name": name,
                "street": street,
                "postal_code": postal,
                "municipality": municipality,
                "latitude": f"{lat:.8f}",
                "longitude": f"{lon:.8f}",
            }
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_route(start_row: dict[str, str], clinic_rows: list[dict[str, str]]) -> list[tuple[float, float]]:
    start = to_latlon(start_row)
    points = [to_latlon(r) for r in clinic_rows]
    return optimize_closed_route(start, points)


def write_path_json(
    path: Path,
    waypoint_loop: list[tuple[float, float]],
    road_path: list[tuple[float, float]],
    waypoint_distance_m: float,
    road_distance_m: float,
) -> None:
    payload = {
        "waypoint_loop": [[lat, lon] for lat, lon in waypoint_loop],
        "road_path": [[lat, lon] for lat, lon in road_path],
        "waypoint_loop_distance_km": round(waypoint_distance_m / 1000.0, 3),
        "road_path_distance_km": round(road_distance_m / 1000.0, 3),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    start_rows = read_csv(START_CSV)
    if not start_rows:
        raise RuntimeError("Missing start point in route_start_point.csv")
    start_row = start_rows[0]

    oslo_rows = read_csv(OSLO_CSV)
    oslo_sandvika_rows = read_csv(OSLO_SANDVIKA_CSV)

    oslo_waypoint_loop = build_route(start_row, oslo_rows)
    oslo_sandvika_waypoint_loop = build_route(start_row, oslo_sandvika_rows)

    # Build one walk graph that covers the larger (Oslo+Sandvika) route, reuse for both.
    walk_graph = build_walk_graph(oslo_sandvika_waypoint_loop)
    oslo_road_path, oslo_road_distance_m = route_on_walk_graph(walk_graph, oslo_waypoint_loop)
    oslo_sandvika_road_path, oslo_sandvika_road_distance_m = route_on_walk_graph(
        walk_graph, oslo_sandvika_waypoint_loop
    )

    write_gpx(OSLO_GPX, oslo_road_path, "Dr.Dropin Oslo Clinics Loop")
    write_gpx(OSLO_SANDVIKA_GPX, oslo_sandvika_road_path, "Dr.Dropin Oslo + Sandvika Clinics Loop")

    write_route_order_csv(OSLO_ORDER_CSV, oslo_waypoint_loop, start_row, oslo_rows)
    write_route_order_csv(
        OSLO_SANDVIKA_ORDER_CSV, oslo_sandvika_waypoint_loop, start_row, oslo_sandvika_rows
    )
    write_path_json(
        OSLO_PATH_JSON,
        oslo_waypoint_loop,
        oslo_road_path,
        route_length_m(oslo_waypoint_loop),
        oslo_road_distance_m,
    )
    write_path_json(
        OSLO_SANDVIKA_PATH_JSON,
        oslo_sandvika_waypoint_loop,
        oslo_sandvika_road_path,
        route_length_m(oslo_sandvika_waypoint_loop),
        oslo_sandvika_road_distance_m,
    )

    summary = {
        "start_point": {
            "street": start_row["street"],
            "postal_code": start_row["postal_code"],
            "municipality": start_row["municipality"],
            "latitude": float(start_row["latitude"]),
            "longitude": float(start_row["longitude"]),
        },
        "oslo": {
            "clinic_count": len(oslo_rows),
            "waypoint_loop_points": len(oslo_waypoint_loop),
            "road_path_points": len(oslo_road_path),
            "waypoint_loop_distance_km": round(route_length_m(oslo_waypoint_loop) / 1000.0, 3),
            "road_path_distance_km": round(oslo_road_distance_m / 1000.0, 3),
            "gpx_file": str(OSLO_GPX.relative_to(ROOT)),
            "order_file": str(OSLO_ORDER_CSV.relative_to(ROOT)),
            "path_file": str(OSLO_PATH_JSON.relative_to(ROOT)),
        },
        "oslo_sandvika": {
            "clinic_count": len(oslo_sandvika_rows),
            "waypoint_loop_points": len(oslo_sandvika_waypoint_loop),
            "road_path_points": len(oslo_sandvika_road_path),
            "waypoint_loop_distance_km": round(route_length_m(oslo_sandvika_waypoint_loop) / 1000.0, 3),
            "road_path_distance_km": round(oslo_sandvika_road_distance_m / 1000.0, 3),
            "gpx_file": str(OSLO_SANDVIKA_GPX.relative_to(ROOT)),
            "order_file": str(OSLO_SANDVIKA_ORDER_CSV.relative_to(ROOT)),
            "path_file": str(OSLO_SANDVIKA_PATH_JSON.relative_to(ROOT)),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote: {OSLO_GPX}")
    print(f"Wrote: {OSLO_SANDVIKA_GPX}")
    print(f"Wrote: {OSLO_ORDER_CSV}")
    print(f"Wrote: {OSLO_SANDVIKA_ORDER_CSV}")
    print(f"Wrote: {OSLO_PATH_JSON}")
    print(f"Wrote: {OSLO_SANDVIKA_PATH_JSON}")
    print(f"Wrote: {SUMMARY_JSON}")
    print(f"Oslo road-path distance: {summary['oslo']['road_path_distance_km']} km")
    print(f"Oslo+Sandvika road-path distance: {summary['oslo_sandvika']['road_path_distance_km']} km")


if __name__ == "__main__":
    main()
