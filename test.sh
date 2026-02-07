#!/usr/bin/env bash
set -euo pipefail

# SSE stream test for route generation.
# Run: bash test.sh
# Optional env overrides:
#   BASE_URL=http://localhost:8080/api
#   CLINIC_COUNT=20
#   RANDOM_STARTS=1200
#   TWO_OPT_ROUNDS=220
#   CLOSE_LOOP=true
#   CLINIC_IDS="id1,id2,id3"
#
# Behavior:
# - With no env overrides, this script auto-picks a heavier subset and dynamic params.
# - If first run gets no preview events (likely cache hit), it retries once with cache-bust params.

BASE_URL="${BASE_URL:-http://localhost:8080/api}"
CLINIC_COUNT="${CLINIC_COUNT:-20}"
CLOSE_LOOP="${CLOSE_LOOP:-true}"
CLINIC_IDS="${CLINIC_IDS:-}"

if [[ -z "${RANDOM_STARTS+x}" ]]; then
  RANDOM_STARTS=$((900 + $(date +%s) % 500))
else
  RANDOM_STARTS="${RANDOM_STARTS}"
fi

if [[ -z "${TWO_OPT_ROUNDS+x}" ]]; then
  TWO_OPT_ROUNDS=$((140 + $(date +%s) % 120))
else
  TWO_OPT_ROUNDS="${TWO_OPT_ROUNDS}"
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Missing dependency: curl"
  exit 1
fi

if [[ "$CLOSE_LOOP" != "true" && "$CLOSE_LOOP" != "false" ]]; then
  echo "CLOSE_LOOP must be true or false (got: $CLOSE_LOOP)"
  exit 1
fi

if ! [[ "$CLINIC_COUNT" =~ ^[0-9]+$ ]] || [[ "$CLINIC_COUNT" -lt 2 ]]; then
  echo "CLINIC_COUNT must be an integer >= 2 (got: $CLINIC_COUNT)"
  exit 1
fi

extract_ids_from_json() {
  local json="$1"
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$json" | jq -r '.clinics[].id'
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    JSON_INPUT="$json" python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["JSON_INPUT"])
for clinic in payload.get("clinics", []):
    cid = clinic.get("id")
    if cid:
        print(cid)
PY
    return
  fi

  if command -v uv >/dev/null 2>&1; then
    JSON_INPUT="$json" UV_CACHE_DIR=.uv-cache uv run --no-project python - <<'PY'
import json
import os

payload = json.loads(os.environ["JSON_INPUT"])
for clinic in payload.get("clinics", []):
    cid = clinic.get("id")
    if cid:
        print(cid)
PY
    return
  fi

  echo "Need jq or python3/uv to parse /api/clinics response."
  exit 1
}

build_ids_json() {
  printf '%s\n' "$@" | awk '
    BEGIN { n = 0; printf "[" }
    NF {
      gsub(/"/, "\\\"");
      if (n++ > 0) printf ",";
      printf "\"%s\"", $0;
    }
    END { printf "]" }
  '
}

declare -a IDS=()
if [[ -n "$CLINIC_IDS" ]]; then
  IFS=',' read -r -a IDS <<<"$CLINIC_IDS"
else
  echo "Fetching clinics from $BASE_URL/clinics ..."
  CLINICS_JSON="$(curl -fsS "$BASE_URL/clinics")"

  declare -a ALL_IDS=()
  while IFS= read -r id; do
    [[ -n "$id" ]] && ALL_IDS+=("$id")
  done < <(extract_ids_from_json "$CLINICS_JSON")

  if [[ "${#ALL_IDS[@]}" -lt 2 ]]; then
    echo "Need at least 2 clinics from API. Found: ${#ALL_IDS[@]}"
    exit 1
  fi

  total="${#ALL_IDS[@]}"
  if [[ "$CLINIC_COUNT" -ge "$total" ]]; then
    IDS=("${ALL_IDS[@]}")
  else
    start_idx=$(( $(date +%s) % total ))
    for ((i = 0; i < CLINIC_COUNT; i++)); do
      idx=$(( (start_idx + i) % total ))
      IDS+=("${ALL_IDS[$idx]}")
    done
  fi
fi

if [[ "${#IDS[@]}" -lt 2 ]]; then
  echo "Need at least 2 clinic IDs. Found: ${#IDS[@]}"
  exit 1
fi

IDS_JSON="$(build_ids_json "${IDS[@]}")"

run_stream_once() {
  local rs="$1"
  local tor="$2"
  local payload
  local tmp_log
  local curl_rc awk_rc

  payload="$(printf '{"clinic_ids":%s,"random_starts":%s,"two_opt_rounds":%s,"close_loop":%s}' \
    "$IDS_JSON" "$rs" "$tor" "$CLOSE_LOOP")"

  echo "Testing SSE stream:"
  echo "  BASE_URL        = $BASE_URL"
  echo "  clinics used    = ${#IDS[@]}"
  echo "  random_starts   = $rs"
  echo "  two_opt_rounds  = $tor"
  echo "  close_loop      = $CLOSE_LOOP"
  echo "  IDs             = ${IDS[*]}"
  echo

  tmp_log="$(mktemp)"
  set +e
  curl -N -sS --no-buffer \
    -H "Accept: text/event-stream" \
    -H "Content-Type: application/json" \
    -X POST "$BASE_URL/route/stream" \
    --data "$payload" \
  | awk -v START_EPOCH="$(date +%s)" '
    BEGIN {
      start = START_EPOCH + 0;
      event = "";
      data = "";
      preview = 0;
      status = 0;
      result = 0;
      err = 0;
    }

    function now_epoch(   cmd, t) {
      cmd = "date +%s";
      cmd | getline t;
      close(cmd);
      return t + 0;
    }

    function json_get(payload, key,   pat, tmp) {
      pat = "\"" key "\"[[:space:]]*:[[:space:]]*";
      if (match(payload, pat) == 0) return "";
      tmp = substr(payload, RSTART + RLENGTH);

      if (substr(tmp, 1, 1) == "\"") {
        tmp = substr(tmp, 2);
        sub(/".*$/, "", tmp);
        return tmp;
      }
      if (match(tmp, /^-?[0-9]+(\.[0-9]+)?/)) {
        return substr(tmp, RSTART, RLENGTH);
      }
      if (substr(tmp, 1, 4) == "true") return "true";
      if (substr(tmp, 1, 5) == "false") return "false";
      return "";
    }

    function flush_event(   t, msg, kind, prog, dist) {
      if (event == "") return;
      t = now_epoch() - start;

      if (event == "preview") {
        preview++;
        kind = json_get(data, "kind");
        prog = json_get(data, "progress_pct");
        dist = json_get(data, "distance_km");
        if (kind == "") kind = "preview";
        if (prog == "") prog = "?";
        if (dist == "") dist = "?";
        printf("[+%3ds] preview #%d kind=%s progress=%s%% dist=%s km\n", t, preview, kind, prog, dist);
        fflush();
      } else if (event == "status") {
        status++;
        msg = json_get(data, "message");
        if (msg != "") {
          printf("[+%3ds] status: %s\n", t, msg);
          fflush();
        }
      } else if (event == "result") {
        result = 1;
        dist = json_get(data, "distance_km");
        if (dist == "") dist = "?";
        printf("[+%3ds] RESULT distance=%s km\n", t, dist);
        fflush();
      } else if (event == "error") {
        err = 1;
        msg = json_get(data, "detail");
        printf("[+%3ds] ERROR %s\n", t, msg);
        fflush();
      }

      event = "";
      data = "";
    }

    {
      gsub(/\r/, "");

      if ($0 ~ /^event:[[:space:]]*/) {
        flush_event();
        event = $0;
        sub(/^event:[[:space:]]*/, "", event);
        next;
      }

      if ($0 ~ /^data:[[:space:]]*/) {
        line = $0;
        sub(/^data:[[:space:]]*/, "", line);
        if (data == "") data = line;
        else data = data "\n" line;
        next;
      }

      if ($0 == "") {
        flush_event();
        next;
      }
    }

    END {
      flush_event();
      print "----";
      printf("Summary: preview=%d status=%d result=%d error=%d\n", preview, status, result, err);
      if (preview == 0) {
        print "No preview events received.";
        exit 2;
      }
      if (err != 0 || result == 0) {
        exit 1;
      }
    }
  ' | tee "$tmp_log"
  local -a ps=("${PIPESTATUS[@]}")
  curl_rc="${ps[0]:-1}"
  awk_rc="${ps[1]:-1}"
  set -e

  if [[ "$curl_rc" -ne 0 ]]; then
    rm -f "$tmp_log"
    return "$curl_rc"
  fi

  if [[ "$awk_rc" -ne 0 ]]; then
    rm -f "$tmp_log"
    return "$awk_rc"
  fi

  rm -f "$tmp_log"
  return 0
}

set +e
run_stream_once "$RANDOM_STARTS" "$TWO_OPT_ROUNDS"
rc="$?"
set -e

if [[ "$rc" -eq 2 ]]; then
  echo
  echo "No preview events on first run (likely cache hit). Retrying once with cache-bust parameters..."
  retry_rs=$((RANDOM_STARTS + 97))
  retry_tor=$((TWO_OPT_ROUNDS + 53))
  run_stream_once "$retry_rs" "$retry_tor"
  exit "$?"
fi

exit "$rc"
