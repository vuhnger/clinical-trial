#!/usr/bin/env python3
"""Generate coordinate verification SVG maps for Dr.Dropin clinic datasets.

No third-party dependencies required.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "art"

OSLO_DATASET = DATA_DIR / "drdropin_clinics_oslo_routing_ready.csv"
OSLO_SANDVIKA_DATASET = DATA_DIR / "drdropin_clinics_oslo_sandvika_routing_ready.csv"
START_DATASET = DATA_DIR / "route_start_point.csv"
LOGO_SVG = ART_DIR / "drd-vector.svg"

OUT_OSLO = ART_DIR / "map_oslo_verification.svg"
OUT_OSLO_SANDVIKA = ART_DIR / "map_oslo_sandvika_verification.svg"


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_point(row: Dict[str, str]) -> Tuple[float, float]:
    return float(row["longitude"]), float(row["latitude"])


def nearest_neighbor_path(
    start: Tuple[float, float], points: Iterable[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Create a simple visual path for map verification, not final routing."""
    remaining = list(points)
    path = [start]
    current = start
    while remaining:
        idx = min(
            range(len(remaining)),
            key=lambda i: (remaining[i][0] - current[0]) ** 2
            + (remaining[i][1] - current[1]) ** 2,
        )
        nxt = remaining.pop(idx)
        path.append(nxt)
        current = nxt
    return path


def load_logo_paths(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"<path\b[^>]*?/?>", text, flags=re.IGNORECASE)


def project(
    lon: float,
    lat: float,
    bounds: Tuple[float, float, float, float],
    width: int,
    height: int,
    margin: int,
) -> Tuple[float, float]:
    min_lon, max_lon, min_lat, max_lat = bounds
    span_lon = max(max_lon - min_lon, 1e-9)
    span_lat = max(max_lat - min_lat, 1e-9)
    x = margin + (lon - min_lon) * (width - 2 * margin) / span_lon
    # SVG Y grows downward, so invert latitude.
    y = margin + (max_lat - lat) * (height - 2 * margin) / span_lat
    return x, y


def compute_bounds(
    points: List[Tuple[float, float]], pad_ratio: float = 0.08
) -> Tuple[float, float, float, float]:
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    lon_pad = max((max_lon - min_lon) * pad_ratio, 0.004)
    lat_pad = max((max_lat - min_lat) * pad_ratio, 0.004)
    return (
        min_lon - lon_pad,
        max_lon + lon_pad,
        min_lat - lat_pad,
        max_lat + lat_pad,
    )


def fmt_points(points_xy: List[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points_xy)


def render_svg(
    title: str,
    clinics: List[Dict[str, str]],
    start: Dict[str, str],
    out_path: Path,
    logo_paths: List[str],
) -> None:
    width = 1600
    height = 1000
    margin = 70

    clinic_points = [to_point(r) for r in clinics]
    start_point = to_point(start)
    all_points = [start_point] + clinic_points
    bounds = compute_bounds(all_points)

    # Visual-only ordering for line preview.
    route_ll = nearest_neighbor_path(start_point, clinic_points)
    route_xy = [project(lon, lat, bounds, width, height, margin) for lon, lat in route_ll]

    clinic_xy = [project(lon, lat, bounds, width, height, margin) for lon, lat in clinic_points]
    start_xy = project(start_point[0], start_point[1], bounds, width, height, margin)

    logo_size = 28
    logo_uses = []
    for (x, y), row in zip(clinic_xy, clinics):
        name = row.get("clinic_name", "Clinic")
        logo_uses.append(
            f'<g><title>{escape_xml(name)}</title>'
            f'<use href="#drdLogo" x="{x - logo_size/2:.2f}" y="{y - logo_size/2:.2f}" '
            f'width="{logo_size}" height="{logo_size}" /></g>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <symbol id="drdLogo" viewBox="0 0 200 200">
      {''.join(logo_paths)}
    </symbol>
  </defs>

  <rect width="{width}" height="{height}" fill="#2F3136" />
  <rect x="{margin}" y="{margin}" width="{width - 2 * margin}" height="{height - 2 * margin}" fill="#3A3D42" rx="20" />

  <text x="{margin}" y="40" fill="#F2F2F2" font-size="30" font-family="Helvetica, Arial, sans-serif">{escape_xml(title)}</text>
  <text x="{margin}" y="72" fill="#D0D0D0" font-size="18" font-family="Helvetica, Arial, sans-serif">Clinics: {len(clinics)} | Start: Sørkedalsveien 8a, 0369 Oslo</text>

  <polyline points="{fmt_points(route_xy)}" fill="none" stroke="#FFD84D" stroke-width="5" stroke-linejoin="round" stroke-linecap="round" opacity="0.95" />

  {''.join(logo_uses)}

  <circle cx="{start_xy[0]:.2f}" cy="{start_xy[1]:.2f}" r="12" fill="#111111" stroke="#FFFFFF" stroke-width="3" />
  <circle cx="{start_xy[0]:.2f}" cy="{start_xy[1]:.2f}" r="4.5" fill="#FFFFFF" />
  <text x="{start_xy[0] + 16:.2f}" y="{start_xy[1] - 16:.2f}" fill="#FFFFFF" font-size="15" font-family="Helvetica, Arial, sans-serif">Start</text>
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def main() -> None:
    oslo_rows = read_rows(OSLO_DATASET)
    oslo_sandvika_rows = read_rows(OSLO_SANDVIKA_DATASET)
    start_rows = read_rows(START_DATASET)
    if not start_rows:
        raise RuntimeError("Missing start point row in route_start_point.csv")
    start = start_rows[0]

    logo_paths = load_logo_paths(LOGO_SVG)
    if not logo_paths:
        raise RuntimeError("No <path/> elements found in art/drd-vector.svg")

    render_svg(
        "Dr.Dropin Coordinate Verification - Oslo",
        oslo_rows,
        start,
        OUT_OSLO,
        logo_paths,
    )
    render_svg(
        "Dr.Dropin Coordinate Verification - Oslo + Sandvika",
        oslo_sandvika_rows,
        start,
        OUT_OSLO_SANDVIKA,
        logo_paths,
    )
    print(f"Generated: {OUT_OSLO}")
    print(f"Generated: {OUT_OSLO_SANDVIKA}")


if __name__ == "__main__":
    main()
