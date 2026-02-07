from __future__ import annotations

from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    clinic_ids: list[str] = Field(default_factory=list)
    random_starts: int | None = None
    two_opt_rounds: int | None = None
    close_loop: bool = False


class Clinic(BaseModel):
    id: str
    navn: str
    gateadresse: str
    postnummer: str
    kommune: str
    omrade: str
    lat: float
    lon: float
