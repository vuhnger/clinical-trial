#!/usr/bin/env python3
"""Generate interactive Leaflet maps for Oslo and Oslo+Sandvika clinic overlays.

Uses only Python stdlib and existing CSV/SVG assets in this repo.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "art"

OSLO_CSV = DATA_DIR / "drdropin_clinics_oslo_routing_ready.csv"
OSLO_SANDVIKA_CSV = DATA_DIR / "drdropin_clinics_oslo_sandvika_routing_ready.csv"
START_CSV = DATA_DIR / "route_start_point.csv"
LOGO_SVG = ART_DIR / "drd-vector.svg"

OUT_OSLO = ART_DIR / "map_oslo_overlay.html"
OUT_OSLO_SANDVIKA = ART_DIR / "map_oslo_sandvika_overlay.html"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_latlon(row: dict[str, str]) -> tuple[float, float]:
    return float(row["latitude"]), float(row["longitude"])


def nearest_neighbor_path(
    start: tuple[float, float], points: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Simple visual ordering (not final road-network route)."""
    remaining = points[:]
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


def same_point(a: tuple[float, float], b: tuple[float, float], eps: float = 1e-7) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def logo_data_uri(svg_path: Path) -> str:
    svg = svg_path.read_text(encoding="utf-8")
    compact = " ".join(svg.split())
    return f"data:image/svg+xml,{quote(compact)}"


def render_map_html(
    title: str,
    subtitle: str,
    rows: list[dict[str, str]],
    start_row: dict[str, str],
    icon_uri: str,
    output_path: Path,
) -> None:
    start_lat, start_lon = to_latlon(start_row)

    clinics = []
    clinic_points = []
    for row in rows:
        lat, lon = to_latlon(row)
        clinic_points.append((lat, lon))
        clinics.append(
            {
                "name": row.get("clinic_name", ""),
                "street": row.get("street", ""),
                "postal_code": row.get("postal_code", ""),
                "municipality": row.get("municipality", ""),
                "lat": lat,
                "lon": lon,
            }
        )

    route_input = [
        point for point in clinic_points if not same_point(point, (start_lat, start_lon))
    ]
    route = nearest_neighbor_path((start_lat, start_lon), route_input)

    all_lats = [start_lat] + [p[0] for p in clinic_points]
    all_lons = [start_lon] + [p[1] for p in clinic_points]
    center_lat = mean(all_lats)
    center_lon = mean(all_lons)

    clinics_js = json.dumps(clinics, ensure_ascii=False)
    route_js = json.dumps(route)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape_html(title)}</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    :root {{
      --bg: #2f3136;
      --panel: #3a3d42;
      --text: #f2f2f2;
      --muted: #c6c8cc;
      --route: #ffd84d;
    }}
    html, body {{
      margin: 0;
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    }}
    .layout {{
      display: grid;
      grid-template-rows: auto 1fr;
      height: 100%;
      gap: 10px;
      padding: 12px;
      box-sizing: border-box;
    }}
    .header {{
      background: var(--panel);
      border-radius: 10px;
      padding: 12px 14px;
    }}
    .title {{
      font-size: 20px;
      font-weight: 700;
      line-height: 1.2;
    }}
    .subtitle {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 14px;
    }}
    #map {{
      width: 100%;
      height: 100%;
      min-height: 680px;
      border-radius: 10px;
      overflow: hidden;
    }}
    .leaflet-popup-content-wrapper,
    .leaflet-popup-tip {{
      background: #2b2d31;
      color: #f4f4f5;
    }}
    .leaflet-container a {{
      color: #ffe27a;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="header">
      <div class="title">{escape_html(title)}</div>
      <div class="subtitle">{escape_html(subtitle)}</div>
    </div>
    <div id="map"></div>
  </div>

  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script>
    const map = L.map('map', {{ zoomControl: true }}).setView([{center_lat}, {center_lon}], 12);

    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      maxZoom: 20,
      subdomains: 'abcd',
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);

    const clinics = {clinics_js};
    const route = {route_js};

    const logoIcon = L.icon({{
      iconUrl: '{icon_uri}',
      iconSize: [24, 24],
      iconAnchor: [12, 12],
      popupAnchor: [0, -12],
      className: 'drdropin-logo-marker'
    }});

    const bounds = [];

    const startMarker = L.circleMarker([{start_lat}, {start_lon}], {{
      radius: 8,
      color: '#ffffff',
      weight: 2,
      fillColor: '#111111',
      fillOpacity: 1
    }}).addTo(map);
    startMarker.bindPopup('<b>Start</b><br/>Sørkedalsveien 8a, 0369 Oslo');
    bounds.push([{start_lat}, {start_lon}]);

    clinics.forEach((clinic) => {{
      const marker = L.marker([clinic.lat, clinic.lon], {{ icon: logoIcon }}).addTo(map);
      marker.bindPopup(
        `<b>${{clinic.name}}</b><br/>${{clinic.street}}, ${{clinic.postal_code}} ${{clinic.municipality}}`
      );
      bounds.push([clinic.lat, clinic.lon]);
    }});

    const routeLine = L.polyline(route, {{
      color: '#ffd84d',
      weight: 4,
      opacity: 0.95,
      lineJoin: 'round'
    }}).addTo(map);

    if (bounds.length > 1) {{
      map.fitBounds(bounds, {{ padding: [24, 24] }});
    }}
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def main() -> None:
    oslo_rows = read_csv(OSLO_CSV)
    oslo_sandvika_rows = read_csv(OSLO_SANDVIKA_CSV)
    start_rows = read_csv(START_CSV)
    if not start_rows:
        raise RuntimeError("Missing route_start_point.csv row")
    start_row = start_rows[0]
    icon_uri = logo_data_uri(LOGO_SVG)

    render_map_html(
        title="Dr.Dropin Coordinate Overlay - Oslo",
        subtitle=f"Clinics: {len(oslo_rows)} | Basemap: CARTO Dark + OpenStreetMap",
        rows=oslo_rows,
        start_row=start_row,
        icon_uri=icon_uri,
        output_path=OUT_OSLO,
    )
    render_map_html(
        title="Dr.Dropin Coordinate Overlay - Oslo + Sandvika",
        subtitle=f"Clinics: {len(oslo_sandvika_rows)} | Basemap: CARTO Dark + OpenStreetMap",
        rows=oslo_sandvika_rows,
        start_row=start_row,
        icon_uri=icon_uri,
        output_path=OUT_OSLO_SANDVIKA,
    )
    print(f"Generated: {OUT_OSLO}")
    print(f"Generated: {OUT_OSLO_SANDVIKA}")


if __name__ == "__main__":
    main()
