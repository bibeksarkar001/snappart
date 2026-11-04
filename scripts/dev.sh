#!/usr/bin/env bash
# One command: LIVE API (:8000, calls Meta, costs money) + RECORDED API (:8200, replay of the part B run, free), their workers and the
# frontend (https :5173). The Live/Recorded switch in the app header picks which API the browser talks to. Ctrl+C stops everything.
# SNAPPART_REPLAY=1 makes the :8000 API recorded too (no key needed). The frontend listens on localhost only; SNAPPART_LAN=1 opens it to
# the network (a phone), but then anyone on that network can spend the live key.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/opt/homebrew/Caskroom/miniforge/base/envs/snappart/bin/python
[ -x "$PY" ] && RUN=("$PY") || RUN=(conda run --no-capture-output -n snappart python)
LOGS="${SNAPPART_DEV_LOGS:-/tmp/snappart-dev}"; mkdir -p "$LOGS/recorded-data"
PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null; exit 0' INT TERM EXIT
live() { (cd "$ROOT/backend" && SNAPPART_REPLAY="${SNAPPART_REPLAY:-0}" exec "${RUN[@]}" "$@") ; }
rec()  { (cd "$ROOT/backend" && SNAPPART_REPLAY=1 SNAPPART_DATA_DIR="$LOGS/recorded-data" exec "${RUN[@]}" "$@") ; }
live -m uvicorn app.main:app --port 8000 >"$LOGS/api.log" 2>&1 & PIDS+=($!)
live -m app.worker >"$LOGS/worker.log" 2>&1 & PIDS+=($!)
rec -m uvicorn app.main:app --port 8200 >"$LOGS/api-recorded.log" 2>&1 & PIDS+=($!)
rec -m app.worker >"$LOGS/worker-recorded.log" 2>&1 & PIDS+=($!)
(cd "$ROOT/frontend" && exec npm run dev >"$LOGS/frontend.log" 2>&1) & PIDS+=($!)
for p in 8000 8200; do
  for _ in $(seq 60); do curl -fs http://127.0.0.1:$p/healthz >/dev/null && break; sleep 1; done
  curl -fs http://127.0.0.1:$p/healthz >/dev/null || { echo "API on :$p did not come up, see $LOGS"; exit 1; }
  echo ":$p $(curl -si http://127.0.0.1:$p/healthz | grep -i '^x-snappart-mode' | tr -d '\r')"
done
echo "SnapPart is up: https://localhost:5173  (Live/Recorded switch in the header; logs in $LOGS, Ctrl+C to stop)"
wait
