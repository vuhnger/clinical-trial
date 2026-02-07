# Dr.Dropin Route Builder

Monorepo med frontend + backend for å:
- velge klinikker direkte i kartet
- beregne kortest mulig løpbar rute langs veier/stier (walk-nettverk)
- vise høydeprofil over kartet
- laste ned GPX

Start/slutt er fast: `Sørkedalsveien 8a, 0369 Oslo`.

## Mappestruktur
- `backend/`: FastAPI API for klinikker, rute og GPX
- `frontend/`: statisk nettside (Leaflet + norsk UI)
- `data/`: klinikkdata
- `output/`: genererte ruter/GPX/cache
- `scripts/`: eksisterende offline-skript
- `docker-compose.yml`: enkel deploy av frontend + backend

## Rask oppstart med Docker (anbefalt)
1. Bygg og start:
```bash
make docker-up
```
Dette skriver også ut lokale URL-er for frontend/backend.

Hvis du kjører `docker compose up` direkte (ikke `-d`), logger containerne:
- `[frontend] UI: http://localhost:8080`
- `[backend] API: http://localhost:8000`

2. Åpne nettsiden:
- `http://localhost:8080`

3. API helse:
- `http://localhost:8000/api/health`

4. Stopp:
```bash
make docker-down
```

## Lokal utvikling uten Docker
1. Installer avhengigheter:
```bash
make sync
```

2. Start backend:
```bash
make backend-dev
```

3. Start frontend (enkel statisk server) i egen terminal:
```bash
python3 -m http.server 8080 --directory frontend
```

4. Åpne:
- `http://localhost:8080`

Merk: Ved lokal frontend via `http.server` må API-kall peke til `http://localhost:8000`.
I Docker-oppsettet håndterer Nginx proxy automatisk `/api`.

## API-endepunkter
- `GET /api/health`
- `GET /api/clinics`
- `POST /api/route`
- `POST /api/route/gpx`

Eksempel request body:
```json
{
  "clinic_ids": ["id-1", "id-2", "id-3"],
  "random_starts": 2500,
  "two_opt_rounds": 400
}
```

## Optimaliseringsparametre
- `random_starts`: antall randomiserte startturer i TSP-søk
- `two_opt_rounds`: dybde på lokal forbedring

Høyere tall kan gi kortere rute, men bruker lengre tid.

## Caching av populære ruter
For å gjøre tunge beregninger raske ved gjentatte kall:
- In-memory LRU-cache for varme ruter (standard `256` entries)
- Persistent disk-cache i `output/api_route_cache/` for restart-sikring
- Popularitetsindeks i `output/api_route_cache/popular_routes.json`

Cache-nøkkel er basert på:
- valgt klinikk-subsett (unik kombinasjon av ID-er)
- `random_starts`
- `two_opt_rounds`

Relevant miljøvariabel:
- `ROUTE_CACHE_MAX_ENTRIES` (standard `256`)

## Viktige filer
### Frontend
- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`

### Backend
- `backend/app/main.py`
- `backend/app/services/routing_service.py`
- `backend/requirements.txt`

### Deploy
- `backend/Dockerfile`
- `frontend/Dockerfile`
- `frontend/nginx.conf`
- `docker-compose.yml`

### Data
- `data/drdropin_clinics_oslo_sandvika_routing_ready.csv`
- `data/route_start_point.csv`

Detaljert datasett-dokumentasjon: `docs/README.md`.
