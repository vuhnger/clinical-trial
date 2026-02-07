# SETUP.md

Denne guiden viser hvordan du:
1. Kloner repoet.
2. Kjører appen lokalt.
3. Setter den opp på en server slik at den er tilgjengelig på internett med HTTPS.

Repo:
`git@github.com:vuhnger/clinical-trial.git`

## 1. Forutsetninger

Du trenger:
- Git
- Docker + Docker Compose plugin
- (Valgfritt for lokal Python-kjøring) `uv` og Python 3.11+
- En Linux-server/VPS for offentlig drift (Ubuntu 22.04+ anbefalt)
- Et domene (for HTTPS)

## 2. Klon repoet

Kjør:

```bash
git clone git@github.com:vuhnger/clinical-trial.git
cd clinical-trial
```

Hvis SSH ikke er konfigurert, bruk HTTPS:

```bash
git clone https://github.com/vuhnger/clinical-trial.git
cd clinical-trial
```

## 3. Lokal kjøring med Docker (anbefalt)

Bygg og start:

```bash
make docker-up
```

Dette starter:
- Frontend på `http://localhost:8080`
- Backend på `http://localhost:8000`

Sjekk health:

```bash
curl http://localhost:8000/api/health
```

Stopp:

```bash
make docker-down
```

## 4. Lokal utvikling uten Docker

Installer dependencies:

```bash
make sync
```

Start backend:

```bash
make backend-dev
```

Start frontend i ny terminal:

```bash
python3 -m http.server 8080 --directory frontend
```

Åpne:
- `http://localhost:8080`

## 5. Produksjon på server (internett-tilgang)

Denne delen er for en VPS med offentlig IP.

## 5.1 Installer Docker på Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
newgrp docker
```

## 5.2 Klon og start appen på server

```bash
git clone git@github.com:vuhnger/clinical-trial.git
cd clinical-trial
docker compose up --build -d
docker compose ps
```

Sjekk lokalt på server:

```bash
curl http://localhost:8000/api/health
curl -I http://localhost:8080
```

## 5.2.1 Anbefalt produksjonsprofil (ikke eksponer app-porter direkte)

Lag `docker-compose.prod.yml`:

```yaml
services:
  backend:
    ports:
      - "127.0.0.1:8000:8000"
  frontend:
    ports:
      - "127.0.0.1:8080:80"
```

Start med prod-overriden:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

Da er `8000` og `8080` kun tilgjengelig fra serveren selv, og publisering skjer via Caddy på 80/443.

## 5.3 DNS (domene)

Opprett DNS-record hos domeneleverandør:
- Type: `A`
- Host: f.eks. `route` (eller `@`)
- Value: serverens offentlige IPv4-adresse

Eksempel slutt-URL:
- `https://route.dittdomene.no`

## 5.4 HTTPS + reverse proxy (Caddy)

Caddy håndterer TLS-sertifikat automatisk (Let’s Encrypt).

Installer Caddy:

```bash
sudo apt-get update
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update
sudo apt-get install -y caddy
```

Lag `/etc/caddy/Caddyfile`:

```caddyfile
route.dittdomene.no {
  reverse_proxy 127.0.0.1:8080
}
```

Aktiver:

```bash
sudo systemctl reload caddy
sudo systemctl status caddy --no-pager
```

Nå skal appen være tilgjengelig på:
- `https://route.dittdomene.no`

## 5.5 Brannmur

Hvis du bruker UFW:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
sudo ufw status
```

Anbefaling:
- Ikke eksponer backend-port `8000` offentlig i produksjon.
- Ikke eksponer frontend-port `8080` offentlig i produksjon.
- Hold kun 80/443 åpne utad.

## 6. Drift og vedlikehold

Oppdatere til ny versjon:

```bash
cd clinical-trial
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d
```

Se logger:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f frontend
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f backend
```

Restart:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart
```

Stoppe:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml down
```

## 7. Ytelse og cache

Backend cacher populære og tidligere beregnede ruter i:
- Minne (LRU)
- Disk: `output/api_route_cache/`

Juster cache-størrelse via env:
- `ROUTE_CACHE_MAX_ENTRIES` (default `256`)

Første ruteberegning for en ny kombinasjon kan være tung.
Neste kall med samme klinikk-subsett + optimizer-parametere går mye raskere.

## 8. Verifisering etter deploy

Sjekk:
1. `https://route.dittdomene.no` åpner frontend.
2. Du kan velge klinikker og generere rute.
3. GPX-knapp laster ned fil.
4. Backend health endpoint virker lokalt på server:
   `curl http://localhost:8000/api/health`

## 9. Vanlige feil

`docker: permission denied`:
- Logg inn på nytt etter `usermod -aG docker $USER`.

`502/Bad Gateway` i Caddy:
- Sjekk at frontend-container kjører:
  `docker compose ps`

Map/rute tar lang tid første gang:
- Forventet ved kald cache og første graf-opplasting.

Ingen HTTPS-sertifikat:
- Kontroller at DNS peker korrekt til serverens offentlige IP.
- Kontroller at port 80/443 er åpne.
