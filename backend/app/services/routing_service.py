from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests


Coordinate = tuple[float, float]  # (lat, lon)
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class ClinicRow:
    id: str
    clinic_name: str
    street: str
    postal_code: str
    municipality: str
    area: str
    country: str
    latitude: float
    longitude: float

    @property
    def point(self) -> Coordinate:
        return (self.latitude, self.longitude)


def same_point(a: Coordinate, b: Coordinate, eps: float = 1e-7) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def haversine_m(a: Coordinate, b: Coordinate) -> float:
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


def path_length_m(path: list[Coordinate]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(haversine_m(path[i], path[i + 1]) for i in range(len(path) - 1))


def _dedupe_points(points: list[Coordinate]) -> list[Coordinate]:
    out: list[Coordinate] = []
    for p in points:
        if not any(same_point(p, q) for q in out):
            out.append(p)
    return out


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _emit_progress(
    progress_callback: ProgressCallback | None, event: str, payload: dict[str, Any]
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(event, payload)
        # Yield briefly so the stream thread can flush events during CPU-heavy work.
        time.sleep(0.001)
    except Exception:
        # Streaming/progress is best-effort and should never break route computation.
        return


def _rotate_tour_start(tour: list[int], start_idx: int = 0) -> list[int]:
    if start_idx not in tour:
        return tour[:]
    pos = tour.index(start_idx)
    return tour[pos:] + tour[:pos]


def _cycle_cost(tour: list[int], dist: list[list[float]]) -> float:
    n = len(tour)
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def _path_cost(path: list[int], dist: list[list[float]]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(dist[path[i]][path[i + 1]] for i in range(len(path) - 1))


def _tour_from_cycle_result(cycle: list[int]) -> list[int]:
    if not cycle:
        return []
    return cycle[:-1] if cycle[0] == cycle[-1] else cycle[:]


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


def _two_opt_cycle(tour: list[int], dist: list[list[float]], max_rounds: int = 200) -> list[int]:
    if len(tour) < 5:
        return _rotate_tour_start(tour, 0)

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


def _two_opt_path(path: list[int], dist: list[list[float]], max_rounds: int = 200) -> list[int]:
    if len(path) < 4:
        return path[:]

    best = path[:]
    best_cost = _path_cost(best, dist)

    improved = True
    rounds = 0
    n = len(best)
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for i in range(0, n - 2):
            for k in range(i + 1, n - 1):
                cand = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                cand_cost = _path_cost(cand, dist)
                if cand_cost + 1e-9 < best_cost:
                    best = cand
                    best_cost = cand_cost
                    improved = True
                    break
            if improved:
                break
    return best


class RoutingService:
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        *,
        default_random_starts: int = 1800,
        default_two_opt_rounds: int = 300,
        route_cache_max_entries: int = 256,
        max_random_starts: int = 1400,
        max_two_opt_rounds: int = 260,
        profile_sample_points: int = 90,
    ) -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.default_random_starts = default_random_starts
        self.default_two_opt_rounds = default_two_opt_rounds
        self.max_random_starts = max(10, max_random_starts)
        self.max_two_opt_rounds = max(10, max_two_opt_rounds)
        self.profile_sample_points = max(20, profile_sample_points)
        self.route_cache_max_entries = max(16, route_cache_max_entries)
        self.disk_cache_max_entries = max(120, self.route_cache_max_entries * 8)

        self._clinics = self._load_clinics()
        self._clinic_by_id = {c.id: c for c in self._clinics}
        self._start = self._load_start()

        self._graph = None
        self._graph_lock = threading.Lock()
        self._routing_cache_lock = threading.Lock()
        self._clinic_node_by_id: dict[str, int] = {}
        self._node_distance_cache: dict[tuple[int, int], float] = {}
        self._node_path_cache: dict[tuple[int, int], list[int]] = {}
        self._cache_lock = threading.Lock()
        self._route_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._disk_cache_dir = self.output_dir / "api_route_cache"
        self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
        self._popularity_path = self._disk_cache_dir / "popular_routes.json"
        self._cache_popularity = self._load_cache_popularity()
        self._popularity_dirty_counter = 0
        self._graph_cache_path = self.output_dir / "oslo_sandvika_walk_network.graphml"

    def _load_clinics(self) -> list[ClinicRow]:
        path = self.data_dir / "drdropin_clinics_oslo_sandvika_routing_ready.csv"
        rows: list[ClinicRow] = []
        id_count: dict[str, int] = {}
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                base_id = _slugify(
                    f"{row['clinic_name']}-{row['street']}-{row['postal_code']}-{row['municipality']}"
                )
                idx = id_count.get(base_id, 0) + 1
                id_count[base_id] = idx
                cid = base_id if idx == 1 else f"{base_id}-{idx}"
                rows.append(
                    ClinicRow(
                        id=cid,
                        clinic_name=row["clinic_name"],
                        street=row["street"],
                        postal_code=row["postal_code"],
                        municipality=row["municipality"],
                        area=row["area"],
                        country=row["country"],
                        latitude=float(row["latitude"]),
                        longitude=float(row["longitude"]),
                    )
                )
        rows.sort(key=lambda c: (c.area, c.municipality, c.clinic_name))
        return rows

    def _load_start(self) -> dict[str, Any]:
        path = self.data_dir / "route_start_point.csv"
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise RuntimeError("Fant ikke startpunkt i route_start_point.csv")
        row = rows[0]
        return {
            "name": row.get("name", "Start"),
            "street": row["street"],
            "postal_code": row["postal_code"],
            "municipality": row["municipality"],
            "country": row["country"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        }

    @property
    def start_point(self) -> Coordinate:
        return (self._start["latitude"], self._start["longitude"])

    def list_clinics(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for c in self._clinics:
            out.append(
                {
                    "id": c.id,
                    "navn": c.clinic_name,
                    "gateadresse": c.street,
                    "postnummer": c.postal_code,
                    "kommune": c.municipality,
                    "omrade": c.area,
                    "lat": c.latitude,
                    "lon": c.longitude,
                }
            )
        return out

    def _ensure_graph(self):
        with self._graph_lock:
            if self._graph is not None:
                return self._graph

            os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))
            os.environ.setdefault("XDG_CACHE_HOME", str((Path.cwd() / ".cache").resolve()))
            Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

            import osmnx as ox

            if self._graph_cache_path.exists():
                self._graph = ox.load_graphml(self._graph_cache_path)
                return self._graph

            points = [c.point for c in self._clinics]
            lats = [p[0] for p in points]
            lons = [p[1] for p in points]
            north = max(lats) + 0.02
            south = min(lats) - 0.02
            east = max(lons) + 0.02
            west = min(lons) - 0.02
            self._graph = ox.graph_from_bbox((west, south, east, north), network_type="walk")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ox.save_graphml(self._graph, self._graph_cache_path)
            return self._graph

    def _ensure_clinic_nodes(self, graph) -> None:
        with self._routing_cache_lock:
            if self._clinic_node_by_id:
                return
            import osmnx as ox

            lats = [c.latitude for c in self._clinics]
            lons = [c.longitude for c in self._clinics]
            nodes = [int(n) for n in ox.distance.nearest_nodes(graph, X=lons, Y=lats)]
            for clinic, node_id in zip(self._clinics, nodes):
                self._clinic_node_by_id[clinic.id] = node_id

    def _get_node_distance(self, graph, u: int, v: int) -> float:
        if u == v:
            return 0.0
        key = (u, v)
        with self._routing_cache_lock:
            cached = self._node_distance_cache.get(key)
        if cached is not None:
            return cached

        import networkx as nx

        try:
            distance = float(nx.shortest_path_length(graph, u, v, weight="length"))
        except Exception:
            distance = 0.0
        with self._routing_cache_lock:
            self._node_distance_cache[key] = distance
        return distance

    def _get_node_path(self, graph, u: int, v: int) -> list[int]:
        if u == v:
            return [u]
        key = (u, v)
        with self._routing_cache_lock:
            cached_path = self._node_path_cache.get(key)
        if cached_path is not None:
            return cached_path

        import networkx as nx

        try:
            path = [int(n) for n in nx.shortest_path(graph, u, v, weight="length")]
        except Exception:
            path = []
        with self._routing_cache_lock:
            self._node_path_cache[key] = path
        return path

    def _optimize_waypoint_loop(
        self,
        graph,
        clinics: list[ClinicRow],
        random_starts: int,
        two_opt_rounds: int,
        close_loop: bool,
        progress_callback: ProgressCallback | None = None,
    ) -> list[Coordinate]:
        import networkx as nx
        self._ensure_clinic_nodes(graph)

        seen_ids: set[str] = set()
        ordered_clinics: list[ClinicRow] = []
        for clinic in clinics:
            if clinic.id not in seen_ids:
                ordered_clinics.append(clinic)
                seen_ids.add(clinic.id)

        if not ordered_clinics:
            return []
        points = [c.point for c in ordered_clinics]
        snapped_nodes = [self._clinic_node_by_id[c.id] for c in ordered_clinics]
        if len(points) == 1:
            return [points[0], points[0]] if close_loop else [points[0]]

        n = len(points)
        approx_dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d_approx = haversine_m(points[i], points[j]) * 1.35
                approx_dist[i][j] = d_approx
                approx_dist[j][i] = d_approx

        # Immediate visual bootstrap so clients can draw a route before heavy graph distances finish.
        bootstrap_rounds = max(6, min(24, two_opt_rounds // 8))
        bootstrap_tour = _nearest_neighbor_tour(0, approx_dist)
        if close_loop:
            bootstrap_tour = _two_opt_cycle(
                bootstrap_tour, approx_dist, max_rounds=bootstrap_rounds
            )
            bootstrap_tour = _rotate_tour_start(bootstrap_tour, 0)
            bootstrap_idx = bootstrap_tour + [bootstrap_tour[0]]
            bootstrap_cost = _cycle_cost(bootstrap_tour, approx_dist)
        else:
            bootstrap_tour = _two_opt_path(
                bootstrap_tour, approx_dist, max_rounds=bootstrap_rounds
            )
            bootstrap_idx = bootstrap_tour[:]
            bootstrap_cost = _path_cost(bootstrap_tour, approx_dist)
        bootstrap_route = [points[i] for i in bootstrap_idx if 0 <= i < n]
        if len(bootstrap_route) >= 2:
            _emit_progress(
                progress_callback,
                "preview",
                {
                    "phase": "distance_matrix",
                    "kind": "bootstrap",
                    "progress_pct": 0.0,
                    "distance_km": round(bootstrap_cost / 1000.0, 3),
                    "route": [{"lat": lat, "lon": lon} for lat, lon in bootstrap_route],
                },
            )

        _emit_progress(
            progress_callback,
            "status",
            {
                "phase": "distance_matrix",
                "message": "Bygger veinett-distanser mellom punkter",
                "sources_total": n,
            },
        )

        # Start from approx distances; progressively replace rows with exact shortest-path lengths.
        dist = [row[:] for row in approx_dist]
        source_report_step = max(1, n // 8)
        refine_preview_step = max(1, n // 6)
        refine_rounds = max(4, min(14, two_opt_rounds // 12))
        for i in range(n):
            ni = snapped_nodes[i]
            try:
                lengths = nx.single_source_dijkstra_path_length(graph, ni, weight="length")
            except Exception:
                lengths = {}

            for j in range(i + 1, n):
                nj = snapped_nodes[j]
                d = float(lengths.get(nj, 0.0))
                if d <= 0.0 and ni != nj:
                    d = approx_dist[i][j]
                dist[i][j] = d
                dist[j][i] = d
                with self._routing_cache_lock:
                    self._node_distance_cache[(ni, nj)] = d
                    self._node_distance_cache[(nj, ni)] = d

            should_report = i == 0 or i == n - 1 or (i + 1) % source_report_step == 0
            if should_report:
                _emit_progress(
                    progress_callback,
                    "status",
                    {
                        "phase": "distance_matrix",
                        "message": f"Beregner veinett-distanser {i + 1}/{n}",
                        "sources_done": i + 1,
                        "sources_total": n,
                        "progress_pct": round(((i + 1) / n) * 100.0, 1),
                    },
                )

            should_preview_refine = (
                i == 0 or i == n - 1 or (i + 1) % refine_preview_step == 0
            )
            if should_preview_refine:
                live_tour = _nearest_neighbor_tour(0, dist)
                if close_loop:
                    live_tour = _two_opt_cycle(
                        live_tour, dist, max_rounds=refine_rounds
                    )
                    live_tour = _rotate_tour_start(live_tour, 0)
                    live_idx = live_tour + [live_tour[0]]
                    live_cost = _cycle_cost(live_tour, dist)
                else:
                    live_tour = _two_opt_path(
                        live_tour, dist, max_rounds=refine_rounds
                    )
                    live_idx = live_tour[:]
                    live_cost = _path_cost(live_tour, dist)
                live_route = [points[k] for k in live_idx if 0 <= k < n]
                if len(live_route) >= 2:
                    _emit_progress(
                        progress_callback,
                        "preview",
                        {
                            "phase": "distance_matrix",
                            "kind": "refine",
                            "sources_done": i + 1,
                            "sources_total": n,
                            "progress_pct": round(((i + 1) / n) * 100.0, 1),
                            "distance_km": round(live_cost / 1000.0, 3),
                            "route": [{"lat": lat, "lon": lon} for lat, lon in live_route],
                        },
                    )

        complete = nx.Graph()
        for i in range(n):
            complete.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                complete.add_edge(i, j, weight=dist[i][j])

        candidates: list[list[int]] = []
        c1 = nx.approximation.traveling_salesman_problem(
            complete,
            cycle=True,
            weight="weight",
            method=nx.approximation.christofides,
        )
        candidates.append(_tour_from_cycle_result(c1))
        c2 = nx.approximation.greedy_tsp(complete, source=0, weight="weight")
        candidates.append(_tour_from_cycle_result(c2))
        candidates.append(_nearest_neighbor_tour(0, dist))

        rng = random.Random(42)
        base = list(range(1, n))
        for _ in range(max(0, random_starts)):
            rng.shuffle(base)
            candidates.append([0] + base[:])

        total_candidates = len(candidates)
        report_step = max(1, total_candidates // 10)
        best_update_count = 0
        _emit_progress(
            progress_callback,
            "status",
            {
                "phase": "optimizing",
                "message": "Optimaliserer rekkefølge mellom punkter",
                "total_candidates": total_candidates,
            },
        )

        best_tour: list[int] | None = None
        best_cost = float("inf")
        for idx, cand in enumerate(candidates, start=1):
            if len(cand) != n or len(set(cand)) != n:
                continue
            if close_loop:
                improved = _two_opt_cycle(cand, dist, max_rounds=two_opt_rounds)
                improved = _rotate_tour_start(improved, 0)
                cost = _cycle_cost(improved, dist)
                preview_idx = improved + [improved[0]] if improved else []
            else:
                improved = _two_opt_path(cand, dist, max_rounds=two_opt_rounds)
                cost = _path_cost(improved, dist)
                preview_idx = improved[:]

            is_best = False
            if cost < best_cost:
                best_cost = cost
                best_tour = improved
                best_update_count += 1
                is_best = True

            if idx == 1 or idx == total_candidates or is_best:
                preview_route = [points[i] for i in preview_idx if 0 <= i < n]
                if len(preview_route) >= 2:
                    _emit_progress(
                        progress_callback,
                        "preview",
                        {
                            "phase": "optimizing",
                            "kind": "candidate",
                            "candidate_index": idx,
                            "total_candidates": total_candidates,
                            "progress_pct": round((idx / total_candidates) * 100.0, 1),
                            "is_best": is_best,
                            "best_update_count": best_update_count,
                            "distance_km": round(cost / 1000.0, 3),
                            "route": [{"lat": lat, "lon": lon} for lat, lon in preview_route],
                        },
                    )
            if idx == 1 or idx == total_candidates or idx % report_step == 0:
                _emit_progress(
                    progress_callback,
                    "status",
                    {
                        "phase": "optimizing",
                        "message": f"Optimalisering {idx}/{total_candidates}",
                        "current_candidate": idx,
                        "total_candidates": total_candidates,
                        "progress_pct": round((idx / total_candidates) * 100.0, 1),
                    },
                )

        if best_tour is None:
            best_tour = _nearest_neighbor_tour(0, dist)

        final_preview_idx = best_tour + [best_tour[0]] if close_loop else best_tour[:]
        final_preview_route = [points[i] for i in final_preview_idx if 0 <= i < n]
        if len(final_preview_route) >= 2:
            _emit_progress(
                progress_callback,
                "preview",
                {
                    "phase": "optimizing",
                    "kind": "best_waypoint_order",
                    "candidate_index": total_candidates,
                    "total_candidates": total_candidates,
                    "progress_pct": 100.0,
                    "is_best": True,
                    "best_update_count": best_update_count,
                    "distance_km": round(best_cost / 1000.0, 3),
                    "route": [{"lat": lat, "lon": lon} for lat, lon in final_preview_route],
                },
            )

        ordered = [points[i] for i in best_tour]
        if close_loop and not same_point(ordered[-1], ordered[0]):
            ordered.append(ordered[0])
        return ordered

    def _edge_length_m(self, graph, u: int, v: int) -> float:
        data = graph.get_edge_data(u, v)
        if not data:
            return 0.0
        lengths = [attrs.get("length", 0.0) for attrs in data.values()]
        return float(min(lengths)) if lengths else 0.0

    def _expand_on_walk_graph(
        self,
        graph,
        waypoint_loop: list[Coordinate],
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[Coordinate], float]:
        import osmnx as ox

        if len(waypoint_loop) < 2:
            return waypoint_loop[:], 0.0

        lats = [p[0] for p in waypoint_loop]
        lons = [p[1] for p in waypoint_loop]
        node_ids = [int(n) for n in ox.distance.nearest_nodes(graph, X=lons, Y=lats)]

        full_nodes: list[int] = []
        total_len = 0.0
        total_segments = len(node_ids) - 1
        report_step = max(1, total_segments // 12)
        _emit_progress(
            progress_callback,
            "status",
            {
                "phase": "building_path",
                "message": "Tracer rute langs gangveier",
                "total_segments": total_segments,
            },
        )
        for i in range(len(node_ids) - 1):
            u = node_ids[i]
            v = node_ids[i + 1]
            if u == v:
                if not full_nodes:
                    full_nodes.append(u)
                continue
            seg = self._get_node_path(graph, u, v)
            if not seg:
                continue
            if full_nodes and full_nodes[-1] == seg[0]:
                full_nodes.extend(seg[1:])
            else:
                full_nodes.extend(seg)

            for j in range(len(seg) - 1):
                total_len += self._edge_length_m(graph, seg[j], seg[j + 1])

            should_report = i == 0 or i == total_segments - 1 or (i + 1) % report_step == 0
            if should_report:
                preview_route = [
                    (float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"])) for n in full_nodes
                ]
                sampled_preview = self._sample_route(preview_route, max_points=380)
                _emit_progress(
                    progress_callback,
                    "preview",
                    {
                        "phase": "building_path",
                        "kind": "network_progress",
                        "segment": i + 1,
                        "total_segments": total_segments,
                        "progress_pct": round(((i + 1) / total_segments) * 100.0, 1),
                        "distance_km": round(total_len / 1000.0, 3),
                        "route": [{"lat": lat, "lon": lon} for lat, lon in sampled_preview],
                    },
                )

        if not full_nodes:
            return waypoint_loop[:], path_length_m(waypoint_loop)
        road_path = [(float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"])) for n in full_nodes]
        return road_path, total_len

    def _sample_route(self, route: list[Coordinate], max_points: int = 120) -> list[Coordinate]:
        if len(route) <= max_points:
            return route[:]
        out: list[Coordinate] = []
        last_idx = len(route) - 1
        for i in range(max_points):
            idx = round(i * last_idx / (max_points - 1))
            out.append(route[idx])
        return out

    def _fetch_elevation(self, points: list[Coordinate]) -> tuple[list[float], str]:
        if not points:
            return [], "none"

        url = "https://api.open-elevation.com/api/v1/lookup"
        locations = [{"latitude": lat, "longitude": lon} for lat, lon in points]
        elevations: list[float] = []
        try:
            chunk_size = 100
            for i in range(0, len(locations), chunk_size):
                chunk = locations[i : i + chunk_size]
                resp = requests.post(url, json={"locations": chunk}, timeout=25)
                resp.raise_for_status()
                payload = resp.json()
                results = payload.get("results", [])
                if len(results) != len(chunk):
                    raise ValueError("Ugyldig høyde-respons")
                elevations.extend(float(item.get("elevation", 0.0)) for item in results)
            return elevations, "open-elevation"
        except Exception:
            return [0.0] * len(points), "fallback-flat"

    def _elevation_totals(
        self, elevations: list[float], *, min_step_m: float = 2.0
    ) -> tuple[int, int]:
        if len(elevations) < 2:
            return 0, 0
        gain = 0.0
        loss = 0.0
        for i in range(1, len(elevations)):
            delta = float(elevations[i]) - float(elevations[i - 1])
            if delta >= min_step_m:
                gain += delta
            elif delta <= -min_step_m:
                loss += -delta
        return int(round(gain)), int(round(loss))

    def _build_profile(
        self, route: list[Coordinate]
    ) -> tuple[list[dict[str, float]], str, int, int]:
        sampled = self._sample_route(route, max_points=self.profile_sample_points)
        elevations, source = self._fetch_elevation(sampled)
        profile: list[dict[str, float]] = []
        km = 0.0
        for i, point in enumerate(sampled):
            if i > 0:
                km += haversine_m(sampled[i - 1], point) / 1000.0
            elev = elevations[i] if i < len(elevations) else 0.0
            profile.append({"km": round(km, 3), "elev": round(float(elev), 1)})
        gain_m, loss_m = self._elevation_totals(elevations)
        return profile, source, gain_m, loss_m

    def _make_gpx(self, route: list[Coordinate], name: str) -> bytes:
        points_xml = "\n".join(
            f'      <trkpt lat="{lat:.8f}" lon="{lon:.8f}"></trkpt>' for lat, lon in route
        )
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="drdropin-route-web" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>{_xml_escape(name)}</name>
    <trkseg>
{points_xml}
    </trkseg>
  </trk>
</gpx>
"""
        return xml.encode("utf-8")

    def _cache_key(
        self, clinic_ids: list[str], random_starts: int, two_opt_rounds: int, close_loop: bool
    ) -> tuple[Any, ...]:
        return (
            "route-v7",
            tuple(sorted(set(clinic_ids))),
            random_starts,
            two_opt_rounds,
            bool(close_loop),
        )

    def _cache_key_hash(self, key: tuple[Any, ...]) -> str:
        raw = json.dumps(key, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _cache_payload_path(self, key: tuple[Any, ...]) -> Path:
        return self._disk_cache_dir / f"route_{self._cache_key_hash(key)}.json"

    def _load_cache_popularity(self) -> dict[str, Any]:
        if not self._popularity_path.exists():
            return {"updated_at_unix": int(time.time()), "routes": {}}
        try:
            payload = json.loads(self._popularity_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Ugyldig format")
            routes = payload.get("routes", {})
            if not isinstance(routes, dict):
                routes = {}
            return {
                "updated_at_unix": int(payload.get("updated_at_unix", int(time.time()))),
                "routes": routes,
            }
        except Exception:
            return {"updated_at_unix": int(time.time()), "routes": {}}

    def _flush_popularity_locked(self) -> None:
        self._cache_popularity["updated_at_unix"] = int(time.time())
        tmp = self._popularity_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._cache_popularity, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._popularity_path)
        self._popularity_dirty_counter = 0

    def _register_cache_access(
        self,
        key: tuple[Any, ...],
        clinic_count: int,
        random_starts: int,
        two_opt_rounds: int,
        cache_layer: str,
        *,
        force_flush: bool = False,
    ) -> None:
        key_hash = self._cache_key_hash(key)
        stat_key = {
            "memory": "memory_hits",
            "disk": "disk_hits",
            "computed": "computed_hits",
        }.get(cache_layer, "computed_hits")
        now_unix = int(time.time())

        with self._cache_lock:
            routes = self._cache_popularity.setdefault("routes", {})
            entry = routes.get(key_hash)
            if not isinstance(entry, dict):
                entry = {
                    "key_hash": key_hash,
                    "cache_file": f"route_{key_hash}.json",
                    "hits": 0,
                    "memory_hits": 0,
                    "disk_hits": 0,
                    "computed_hits": 0,
                    "clinic_count": clinic_count,
                    "random_starts": random_starts,
                    "two_opt_rounds": two_opt_rounds,
                    "created_at_unix": now_unix,
                }
            entry["hits"] = int(entry.get("hits", 0)) + 1
            entry[stat_key] = int(entry.get(stat_key, 0)) + 1
            entry["clinic_count"] = clinic_count
            entry["random_starts"] = random_starts
            entry["two_opt_rounds"] = two_opt_rounds
            entry["last_accessed_unix"] = now_unix
            routes[key_hash] = entry

            self._popularity_dirty_counter += 1
            if force_flush or self._popularity_dirty_counter >= 5:
                self._flush_popularity_locked()

    def _get_cached_memory(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        with self._cache_lock:
            payload = self._route_cache.get(key)
            if payload is None:
                return None
            self._route_cache.move_to_end(key)
            return payload

    def _set_cached_memory(self, key: tuple[Any, ...], payload: dict[str, Any]) -> None:
        with self._cache_lock:
            self._route_cache[key] = payload
            self._route_cache.move_to_end(key)
            while len(self._route_cache) > self.route_cache_max_entries:
                self._route_cache.popitem(last=False)

    def _get_cached_disk(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        path = self._cache_payload_path(key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("Ugyldig cache-objekt")

            # New cache format: {"cache_key": [...], "payload": {...}}
            if "payload" in raw:
                payload = raw.get("payload")
                if not isinstance(payload, dict):
                    raise ValueError("Ugyldig payload")
                return payload

            # Backward compatible format where payload was stored directly.
            if "route" in raw and "distance_km" in raw:
                return raw
            raise ValueError("Ukjent cache-format")
        except Exception:
            try:
                path.unlink()
            except Exception:
                pass
            return None

    def _compact_disk_cache(self) -> None:
        files = sorted(
            self._disk_cache_dir.glob("route_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if len(files) <= self.disk_cache_max_entries:
            return
        for stale in files[self.disk_cache_max_entries :]:
            try:
                stale.unlink()
            except Exception:
                pass

    def _set_cached_disk(self, key: tuple[Any, ...], payload: dict[str, Any]) -> None:
        path = self._cache_payload_path(key)
        wrapped = {
            "cache_version": 2,
            "saved_at_unix": int(time.time()),
            "cache_key": list(key),
            "payload": payload,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(wrapped, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        self._compact_disk_cache()

    def compute_route(
        self,
        clinic_ids: list[str],
        *,
        random_starts: int | None = None,
        two_opt_rounds: int | None = None,
        close_loop: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        selected_ids = list(dict.fromkeys(clinic_ids))
        if len(selected_ids) < 2:
            raise ValueError("Velg minst to klinikker.")

        unknown = [cid for cid in selected_ids if cid not in self._clinic_by_id]
        if unknown:
            raise ValueError(f"Ukjente klinikk-IDer: {', '.join(unknown)}")

        canonical_ids = sorted(set(selected_ids))
        requested_rs = random_starts or self.default_random_starts
        requested_tor = two_opt_rounds or self.default_two_opt_rounds
        rs = min(requested_rs, self.max_random_starts)
        tor = min(requested_tor, self.max_two_opt_rounds)
        key = self._cache_key(canonical_ids, rs, tor, close_loop)

        cached = self._get_cached_memory(key)
        if cached is not None:
            _emit_progress(
                progress_callback,
                "status",
                {"phase": "cache_hit", "layer": "memory", "message": "Rute funnet i minne-cache"},
            )
            self._register_cache_access(
                key, len(canonical_ids), rs, tor, "memory", force_flush=False
            )
            return cached

        cached = self._get_cached_disk(key)
        if cached is not None:
            _emit_progress(
                progress_callback,
                "status",
                {"phase": "cache_hit", "layer": "disk", "message": "Rute funnet i disk-cache"},
            )
            self._set_cached_memory(key, cached)
            self._register_cache_access(
                key, len(canonical_ids), rs, tor, "disk", force_flush=False
            )
            return cached

        _emit_progress(
            progress_callback,
            "status",
            {"phase": "graph_loading", "message": "Laster veinett og forbereder beregning"},
        )
        selected_clinics = [self._clinic_by_id[cid] for cid in canonical_ids]

        graph = self._ensure_graph()
        waypoint_loop = self._optimize_waypoint_loop(
            graph,
            selected_clinics,
            random_starts=rs,
            two_opt_rounds=tor,
            close_loop=close_loop,
            progress_callback=progress_callback,
        )
        road_path, road_distance_m = self._expand_on_walk_graph(
            graph, waypoint_loop, progress_callback=progress_callback
        )
        _emit_progress(
            progress_callback,
            "status",
            {"phase": "elevation", "message": "Henter høydeprofil"},
        )
        profile, elev_source, elevation_gain_m, elevation_loss_m = self._build_profile(road_path)

        payload = {
            "tittel": "Løperute mellom Dr. Dropin klinikker",
            "clinic_count": len(selected_clinics),
            "distance_km": round(road_distance_m / 1000.0, 3),
            "route": [{"lat": lat, "lon": lon} for lat, lon in road_path],
            "profile": profile,
            "elevation_source": elev_source,
            "elevation_gain_m": elevation_gain_m,
            "elevation_loss_m": elevation_loss_m,
            "selected_clinics": [{"id": c.id, "navn": c.clinic_name} for c in selected_clinics],
            "close_loop": bool(close_loop),
            "optimizer": {"random_starts": rs, "two_opt_rounds": tor},
        }
        self._set_cached_memory(key, payload)
        self._set_cached_disk(key, payload)
        self._register_cache_access(key, len(canonical_ids), rs, tor, "computed", force_flush=True)
        _emit_progress(
            progress_callback,
            "status",
            {
                "phase": "done",
                "message": "Rute ferdig beregnet",
                "distance_km": payload["distance_km"],
            },
        )
        return payload

    def build_gpx(
        self,
        clinic_ids: list[str],
        *,
        random_starts: int | None = None,
        two_opt_rounds: int | None = None,
        close_loop: bool = False,
    ) -> tuple[bytes, str]:
        route_payload = self.compute_route(
            clinic_ids,
            random_starts=random_starts,
            two_opt_rounds=two_opt_rounds,
            close_loop=close_loop,
        )
        route = [(point["lat"], point["lon"]) for point in route_payload["route"]]
        gpx = self._make_gpx(route, "Løperute mellom Dr. Dropin klinikker")
        filename = "drdropin-loperute.gpx"
        return gpx, filename
