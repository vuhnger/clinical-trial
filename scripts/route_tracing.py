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
import random
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


def _rotate_cycle_to_start(cycle_nodes: list[int], start_idx: int) -> list[int]:
    if not cycle_nodes:
        return []
    pos = cycle_nodes.index(start_idx)
    return cycle_nodes[pos:] + cycle_nodes[:pos]


def _rotate_tour_start(tour: list[int], start_idx: int = 0) -> list[int]:
    if start_idx not in tour:
        return tour[:]
    pos = tour.index(start_idx)
    return tour[pos:] + tour[:pos]


def _cycle_cost(tour: list[int], dist: list[list[float]]) -> float:
    n = len(tour)
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def _two_opt_cycle(tour: list[int], dist: list[list[float]], max_rounds: int = 80) -> list[int]:
    """2-opt local search for cycle with fixed start at index 0."""
    if len(tour) < 5:
        return tour[:]

    best = _rotate_tour_start(tour, 0)
    best_cost = _cycle_cost(best, dist)

    improved = True
    rounds = 0
    n = len(best)
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                cand = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                cand_cost = _cycle_cost(cand, dist)
                if cand_cost + 1e-9 < best_cost:
                    best = cand
                    best_cost = cand_cost
                    improved = True
                    break
            if improved:
                break
    return best


def _nearest_neighbor_tour(start_idx: int, dist: list[list[float]]) -> list[int]:
    n = len(dist)
    remaining = set(range(n))
    remaining.remove(start_idx)
    tour = [start_idx]
    current = start_idx
    while remaining:
        nxt = min(remaining, key=lambda j: dist[current][j])
        tour.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return tour


def _tour_from_cycle_result(cycle: list[int]) -> list[int]:
    if not cycle:
        return []
    return cycle[:-1] if cycle[0] == cycle[-1] else cycle[:]


def optimize_closed_route_on_graph(
    graph,
    start: Coordinate,
    waypoints: Iterable[Coordinate],
    *,
    random_starts: int = 400,
    two_opt_rounds: int = 120,
    random_seed: int = 42,
) -> list[Coordinate]:
    """Optimize a closed waypoint loop using road-network distances.

    Uses multiple candidate tours + 2-opt on the complete road-distance graph
    and returns the best cycle anchored at the start point.
    """
    import networkx as nx
    import osmnx as ox

    cleaned = [p for p in _dedupe_points(waypoints) if not same_point(p, start)]
    if not cleaned:
        return [start, start]

    points = [start] + cleaned
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    snapped_nodes = [int(n) for n in ox.distance.nearest_nodes(graph, X=lons, Y=lats)]

    unique_nodes = sorted(set(snapped_nodes))
    dijkstra_cache: dict[int, dict[int, float]] = {
        node: nx.single_source_dijkstra_path_length(graph, node, weight="length")
        for node in unique_nodes
    }

    n = len(points)
    dist = [[0.0] * n for _ in range(n)]
    complete = nx.Graph()
    for i in range(n):
        complete.add_node(i)

    for i in range(n):
        for j in range(i + 1, n):
            ni = snapped_nodes[i]
            nj = snapped_nodes[j]
            if ni == nj:
                d = 0.0
            else:
                d = float(dijkstra_cache[ni].get(nj, haversine_m(points[i], points[j]) * 1.5))
            dist[i][j] = d
            dist[j][i] = d
            complete.add_edge(i, j, weight=d)

    candidates: list[list[int]] = []

    # 1) Christofides cycle.
    c1 = nx.approximation.traveling_salesman_problem(
        complete,
        cycle=True,
        weight="weight",
        method=nx.approximation.christofides,
    )
    candidates.append(_tour_from_cycle_result(c1))

    # 2) Greedy cycle.
    c2 = nx.approximation.greedy_tsp(complete, source=0, weight="weight")
    candidates.append(_tour_from_cycle_result(c2))

    # 3) Nearest-neighbor seed.
    candidates.append(_nearest_neighbor_tour(0, dist))

    # 4) Multi-start shuffled seeds for extra local minima exploration.
    rng = random.Random(random_seed)
    base = list(range(1, n))
    for _ in range(max(0, random_starts)):
        rng.shuffle(base)
        candidates.append([0] + base[:])

    best_tour: list[int] | None = None
    best_cost = float("inf")
    for cand in candidates:
        if not cand:
            continue
        # Ensure all nodes exactly once.
        cand_set = set(cand)
        if len(cand) != n or len(cand_set) != n:
            continue
        improved = _two_opt_cycle(cand, dist, max_rounds=two_opt_rounds)
        improved = _rotate_tour_start(improved, 0)
        cost = _cycle_cost(improved, dist)
        if cost < best_cost:
            best_cost = cost
            best_tour = improved

    if best_tour is None:
        best_tour = _nearest_neighbor_tour(0, dist)

    ordered = [points[i] for i in best_tour]
    if not same_point(ordered[0], start):
        ordered = [start] + ordered
    if not same_point(ordered[-1], start):
        ordered.append(start)
    return ordered


def build_walk_graph(
    points: Iterable[Coordinate], padding_deg: float = 0.02, cache_path: Path | None = None
):
    """Build an OSM walk graph for the bounding box covering provided points."""
    # Avoid unwritable cache directories in restricted environments.
    os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str((Path.cwd() / ".cache").resolve()))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    import osmnx as ox

    if cache_path is not None and cache_path.exists():
        return ox.load_graphml(cache_path)

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
    graph = ox.graph_from_bbox((west, south, east, north), network_type="walk", simplify=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, cache_path)
    return graph


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
