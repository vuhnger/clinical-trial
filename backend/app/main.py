from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from backend.app.models import RouteRequest
from backend.app.services.routing_service import RoutingService


DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", Path(__file__).resolve().parents[2] / "output"))
DEFAULT_RANDOM_STARTS = int(os.getenv("ROUTE_RANDOM_STARTS", "900"))
DEFAULT_TWO_OPT_ROUNDS = int(os.getenv("ROUTE_TWO_OPT_ROUNDS", "140"))
ROUTE_CACHE_MAX_ENTRIES = int(os.getenv("ROUTE_CACHE_MAX_ENTRIES", "256"))
MAX_RANDOM_STARTS = int(os.getenv("ROUTE_MAX_RANDOM_STARTS", "1400"))
MAX_TWO_OPT_ROUNDS = int(os.getenv("ROUTE_MAX_TWO_OPT_ROUNDS", "260"))
PROFILE_SAMPLE_POINTS = int(os.getenv("PROFILE_SAMPLE_POINTS", "90"))

service = RoutingService(
    DATA_DIR,
    OUTPUT_DIR,
    default_random_starts=DEFAULT_RANDOM_STARTS,
    default_two_opt_rounds=DEFAULT_TWO_OPT_ROUNDS,
    route_cache_max_entries=ROUTE_CACHE_MAX_ENTRIES,
    max_random_starts=MAX_RANDOM_STARTS,
    max_two_opt_rounds=MAX_TWO_OPT_ROUNDS,
    profile_sample_points=PROFILE_SAMPLE_POINTS,
)

app = FastAPI(title="Dr.Dropin Rute API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/clinics")
def clinics() -> JSONResponse:
    return JSONResponse({"clinics": service.list_clinics()})


@app.post("/api/route")
def route_preview(payload: RouteRequest) -> JSONResponse:
    try:
        result = service.compute_route(
            payload.clinic_ids,
            random_starts=payload.random_starts,
            two_opt_rounds=payload.two_opt_rounds,
            close_loop=payload.close_loop,
        )
        return JSONResponse(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Kunne ikke beregne rute: {exc}") from exc


@app.post("/api/route/gpx")
def route_gpx(payload: RouteRequest) -> Response:
    try:
        gpx_bytes, filename = service.build_gpx(
            payload.clinic_ids,
            random_starts=payload.random_starts,
            two_opt_rounds=payload.two_opt_rounds,
            close_loop=payload.close_loop,
        )
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(content=gpx_bytes, media_type="application/gpx+xml", headers=headers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Kunne ikke lage GPX: {exc}") from exc
