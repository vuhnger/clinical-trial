#!/usr/bin/env python3
"""Route tracing utilities for fixed-start closed-loop route optimization.

Algorithm:
- Seed with nearest-neighbor from the fixed start.
- Improve with 2-opt local search.
- Close the route by returning to the fixed start.
- Expand waypoint loops to road-following paths on an OSM walk network.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable


Coordinate = tuple[float, float]  # (lat, lon)


def same_point(a: Coordinate, b: Coordinate, eps: float = 1e-7) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def haversine_m(a: Coordinate, b: Coordinate) -> float:
    """Great-circle distance in meters."""
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    aa = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(aa), math.sqrt(1 - aa))


def route_length_m(route: Iterable[Coordinate]) -> float:
    pts = list(route)
    if len(pts) < 2:
        return 0.0
    return sum(haversine_m(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def _dedupe_points(points: Iterable[Coordinate], eps: float = 1e-7) -> list[Coordinate]:
    unique: list[Coordinate] = []
    for p in points:
        if not any(same_point(p, q, eps=eps) for q in unique):
            unique.append(p)
    return unique


def _nearest_neighbor_seed(start: Coordinate, waypoints: list[Coordinate]) -> list[Coordinate]:
    remaining = waypoints[:]
    route = [start]
    current = start
    while remaining:
        idx = min(range(len(remaining)), key=lambda i: haversine_m(current, remaining[i]))
        nxt = remaining.pop(idx)
        route.append(nxt)
        current = nxt
    return route


def _two_opt_open(route: list[Coordinate], max_rounds: int = 25) -> list[Coordinate]:
    """2-opt for open path with fixed route[0] (start)."""
    if len(route) < 4:
        return route

    best = route[:]
    best_len = route_length_m(best)

    rounds = 0
    improved = True
    while improved and rounds < max_rounds:
        rounds += 1
        improved = False
        n = len(best)
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                cand = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                cand_len = route_length_m(cand)
                if cand_len + 1e-6 < best_len:
                    best = cand
                    best_len = cand_len
                    improved = True
                    break
            if improved:
                break

    return best


def optimize_open_route(start: Coordinate, waypoints: Iterable[Coordinate]) -> list[Coordinate]:
    """Return an optimized open route [start, ..., end], start fixed, no duplicates."""
    cleaned = [p for p in _dedupe_points(waypoints) if not same_point(p, start)]
    seed = _nearest_neighbor_seed(start, cleaned)
    return _two_opt_open(seed)


def optimize_closed_route(start: Coordinate, waypoints: Iterable[Coordinate]) -> list[Coordinate]:
    """Return an optimized closed route [start, ..., start], start fixed.

    The returned route includes the start coordinate exactly twice:
    - index 0
    - final index (closing the loop)
    """
    open_route = optimize_open_route(start, waypoints)
    if not same_point(open_route[-1], start):
        return open_route + [start]
    return open_route


def build_walk_graph(points: Iterable[Coordinate], padding_deg: float = 0.02):
    """Build an OSM walk graph for the bounding box covering provided points."""
    # Avoid unwritable cache directories in restricted environments.
    os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str((Path.cwd() / ".cache").resolve()))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    import osmnx as ox

    pts = list(points)
    if not pts:
        raise ValueError("Cannot build graph for empty points.")

    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    north = max(lats) + padding_deg
    south = min(lats) - padding_deg
    east = max(lons) + padding_deg
    west = min(lons) - padding_deg

    # OSMnx expects bbox=(left, bottom, right, top) i.e. (west, south, east, north).
    return ox.graph_from_bbox((west, south, east, north), network_type="walk", simplify=True)


def _edge_length_m(graph, u: int, v: int) -> float:
    data = graph.get_edge_data(u, v)
    if not data:
        return 0.0
    lengths = [attrs.get("length", 0.0) for attrs in data.values()]
    return float(min(lengths)) if lengths else 0.0


def route_on_walk_graph(graph, waypoint_loop: list[Coordinate]) -> tuple[list[Coordinate], float]:
    """Expand waypoint loop into a road-following coordinate path and total length."""
    import networkx as nx
    import osmnx as ox

    if len(waypoint_loop) < 2:
        return waypoint_loop[:], 0.0

    lats = [p[0] for p in waypoint_loop]
    lons = [p[1] for p in waypoint_loop]
    nodes = ox.distance.nearest_nodes(graph, X=lons, Y=lats)
    node_ids = list(nodes)

    full_nodes: list[int] = []
    total_len = 0.0
    for i in range(len(node_ids) - 1):
        u = int(node_ids[i])
        v = int(node_ids[i + 1])
        if u == v:
            if not full_nodes:
                full_nodes.append(u)
            continue

        seg_nodes = nx.shortest_path(graph, u, v, weight="length")
        if not seg_nodes:
            continue
        if full_nodes and full_nodes[-1] == seg_nodes[0]:
            full_nodes.extend(seg_nodes[1:])
        else:
            full_nodes.extend(seg_nodes)

    if not full_nodes:
        # Fallback: return waypoint loop if graph routing fails unexpectedly.
        return waypoint_loop[:], route_length_m(waypoint_loop)

    for i in range(len(full_nodes) - 1):
        total_len += _edge_length_m(graph, full_nodes[i], full_nodes[i + 1])

    road_path = [(float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"])) for n in full_nodes]
    return road_path, total_len


def write_gpx(path: Path, route: list[Coordinate], track_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points_xml = "\n".join(
        f'      <trkpt lat="{lat:.8f}" lon="{lon:.8f}"></trkpt>' for lat, lon in route
    )
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="drdropin-route-builder" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>{_xml_escape(track_name)}</name>
    <trkseg>
{points_xml}
    </trkseg>
  </trk>
</gpx>
"""
    path.write_text(xml, encoding="utf-8")


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
