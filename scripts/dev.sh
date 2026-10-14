#!/usr/bin/env bash
# Depth Ledger local stack: boot / down / restart / status
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
LOG_DIR="${LOG_DIR:-$ROOT/.logs}"
API_LOG="$LOG_DIR/api.log"
WEB_LOG="$LOG_DIR/web.log"
PID_DIR="$LOG_DIR"
API_PID_FILE="$PID_DIR/api.pid"
WEB_PID_FILE="$PID_DIR/web.pid"
WEB_PORT_FILE="$PID_DIR/web.port"
VITE_BIN="$ROOT/web/node_modules/.bin/vite"

VENV_BIN="$ROOT/.venv/bin"
if [[ -x "$VENV_BIN/uvicorn" ]]; then
  UVICORN="$VENV_BIN/uvicorn"
  PYTHON="$VENV_BIN/python"
else
  UVICORN="uvicorn"
  PYTHON="python3"
fi

# Fast boot while iterating on routes/UI (model loads on first generate).
SKIP_WARMUP="${DEPTH_LEDGER_SKIP_WARMUP:-0}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

  boot       Free ports if needed, start API + Vite (idempotent if healthy)
  restart    Stop then boot
  down       Stop API + Vite for this project
  status     Show listeners + /api/status warmup

Env:
  API_PORT / WEB_PORT          defaults 8000 / 5173
  DEPTH_LEDGER_SKIP_WARMUP=1   skip Depth Pro load at startup
  LOG_DIR                      default $ROOT/.logs
EOF
}

mkdir -p "$LOG_DIR"

load_stored_web_port() {
  if [[ -f "$WEB_PORT_FILE" ]]; then
    WEB_PORT="$(tr -d '[:space:]' <"$WEB_PORT_FILE")"
  fi
}

save_web_port() {
  echo "$WEB_PORT" >"$WEB_PORT_FILE"
}

pids_on_port() {
  local port="$1"
  lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
}

kill_pids() {
  local pids="$1"
  [[ -z "$pids" ]] && return 0
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 0.4
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null || true
}

free_port() {
  local port="$1"
  local pids
  pids="$(pids_on_port "$port")"
  if [[ -n "$pids" ]]; then
    echo "Freeing :$port (pids: $pids)"
    kill_pids "$pids"
  fi
}

stop_stack() {
  echo "Stopping Depth Ledger…"
  load_stored_web_port
  free_port "$API_PORT"
  free_port "$WEB_PORT"
  free_port 5173
  free_port 5174
  # Orphans that lost the listen socket but still hold RAM (warmup OOMs).
  pkill -f "uvicorn server.main:app" 2>/dev/null || true
  pkill -f "${ROOT}/web/node_modules/.bin/vite" 2>/dev/null || true
  pkill -f "vite --host 127.0.0.1 --port ${WEB_PORT}" 2>/dev/null || true
  # Another project's Vite on the same port (often ::1 only) breaks http://127.0.0.1:5173/
  free_foreign_web
  rm -f "$API_PID_FILE" "$WEB_PID_FILE"
  sleep 0.3
}

port_listening() {
  local port="$1"
  [[ -n "$(pids_on_port "$port")" ]]
}

api_healthy() {
  curl -fsS --connect-timeout 1 --max-time 2 \
    "http://127.0.0.1:${API_PORT}/api/status" >/dev/null 2>&1
}

web_healthy() {
  curl -fsS --connect-timeout 1 --max-time 2 \
    -o /dev/null "http://127.0.0.1:${WEB_PORT}/" 2>/dev/null
}

web_listener_ours() {
  local pids pid args
  pids="$(pids_on_port "$WEB_PORT")"
  [[ -z "$pids" ]] && return 1
  pid="${pids%%$'\n'*}"
  args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  [[ "$args" == *"$ROOT/web"* ]]
}

free_foreign_web() {
  if web_listener_ours; then
    return 0
  fi
  local pids
  pids="$(pids_on_port "$WEB_PORT")"
  if [[ -n "$pids" ]]; then
    echo "Freeing :$WEB_PORT (foreign Vite — not $ROOT/web)" >&2
    kill_pids "$pids"
  fi
}

dump_log_tail() {
  local log="$1"
  if [[ -f "$log" ]]; then
    echo "---- last lines: $log ----" >&2
    tail -n 40 "$log" >&2 || true
  fi
}

wait_port() {
  local name="$1"
  local port="$2"
  local tries="${3:-40}"
  local log="${4:-}"
  local i=0
  while (( i < tries )); do
    if port_listening "$port"; then
      echo "$name ready"
      return 0
    fi
    sleep 0.5
    i=$((i + 1))
  done
  echo "$name did not become ready within $((tries / 2))s" >&2
  [[ -n "$log" ]] && dump_log_tail "$log"
  return 1
}

wait_healthy() {
  local name="$1"
  local fn="$2"
  local tries="${3:-60}"
  local log="${4:-}"
  local i=0
  while (( i < tries )); do
    if "$fn"; then
      echo "$name ready"
      return 0
    fi
    sleep 0.5
    i=$((i + 1))
  done
  echo "$name did not become ready within $((tries / 2))s" >&2
  [[ -n "$log" ]] && dump_log_tail "$log"
  return 1
}

start_api() {
  if api_healthy; then
    echo "API already up on :$API_PORT"
    return 0
  fi
  free_port "$API_PORT"
  echo "Starting API on :$API_PORT (log: $API_LOG)"
  nohup env PYTHONPATH=. DEPTH_LEDGER_SKIP_WARMUP="$SKIP_WARMUP" \
    "$UVICORN" server.main:app --reload \
    --reload-dir server --reload-dir depth_core --reload-dir modal_app \
    --host 127.0.0.1 --port "$API_PORT" >>"$API_LOG" 2>&1 &
  echo $! >"$API_PID_FILE"
  disown -h "$!" 2>/dev/null || true
}

start_web() {
  if web_healthy && web_listener_ours; then
    echo "Web already up on :$WEB_PORT"
    return 0
  fi
  free_foreign_web
  free_port "$WEB_PORT"
  if [[ ! -x "$VITE_BIN" ]]; then
    echo "Missing $VITE_BIN — run: make install" >&2
    exit 1
  fi
  echo "Starting Vite on :$WEB_PORT (log: $WEB_LOG)"
  (
    cd "$ROOT/web"
    nohup "$VITE_BIN" --host 127.0.0.1 --port "$WEB_PORT" >>"$WEB_LOG" 2>&1 &
    echo $! >"$WEB_PID_FILE"
  )
}

pick_web_port() {
  local try="$WEB_PORT"
  local alt=5174
  [[ "$try" == "5174" ]] && alt=5175
  if port_listening "$try" && ! web_listener_ours; then
    echo "Port :$try is used by another dev server — using :$alt for Depth Ledger" >&2
    WEB_PORT="$alt"
  fi
}

cmd_boot() {
  pick_web_port
  save_web_port
  export WEB_PORT
  start_api
  start_web
  # Bind first (reloader can print "running" before the worker listens).
  wait_port "API listen :$API_PORT" "$API_PORT" 80 "$API_LOG"
  wait_port "Web listen :$WEB_PORT" "$WEB_PORT" 40 "$WEB_LOG"
  wait_healthy "API HTTP" api_healthy 90 "$API_LOG"
  if ! wait_healthy "Web HTTP" web_healthy 40 "$WEB_LOG"; then
    exit 1
  fi
  if ! web_listener_ours; then
    echo "Port :$WEB_PORT is not served by $ROOT/web (another Vite may have taken it)." >&2
    dump_log_tail "$WEB_LOG"
    exit 1
  fi
  echo
  echo "UI  http://127.0.0.1:${WEB_PORT}/"
  echo "API http://127.0.0.1:${API_PORT}/api/status"
  if [[ "$SKIP_WARMUP" == "1" || "$SKIP_WARMUP" == "true" || "$SKIP_WARMUP" == "yes" ]]; then
    echo "(warmup skipped — model loads on first generate)"
  fi
}

cmd_status() {
  load_stored_web_port
  local ap wp
  ap="$(pids_on_port "$API_PORT")"
  wp="$(pids_on_port "$WEB_PORT")"
  echo "API :$API_PORT  listen=${ap:-none}  healthy=$(api_healthy && echo yes || echo no)"
  echo "Web :$WEB_PORT  listen=${wp:-none}  healthy=$(web_healthy && echo yes || echo no)"
  if api_healthy; then
    "$PYTHON" - <<PY
import json, urllib.request
u = urllib.request.urlopen("http://127.0.0.1:${API_PORT}/api/status", timeout=3)
d = json.load(u)
api = d.get("api") or {}
inf = d.get("local_infer") or {}
print(f"warmup={api.get('warmup')}  infer={inf.get('state')}  device={inf.get('device')}")
PY
  fi
}

cmd="${1:-}"
case "$cmd" in
  boot|up|start) cmd_boot ;;
  restart|reboot) stop_stack; cmd_boot ;;
  down|stop) stop_stack; echo "Stopped." ;;
  status) cmd_status ;;
  -h|--help|help|"") usage; [[ -n "$cmd" ]] || exit 1 ;;
  *) echo "Unknown command: $cmd" >&2; usage; exit 1 ;;
esac
