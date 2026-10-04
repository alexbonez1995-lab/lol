#!/usr/bin/env bash
# OBITO v4.0 – Starter für Linux/macOS: Ollama prüfen/starten, Systemcheck, Server, HUD öffnen.
set -u
cd "$(dirname "$0")"

OBITO_URL="http://127.0.0.1:8765"
OLLAMA_URL="http://127.0.0.1:11434"
PY="${PYTHON:-python3}"

ok()   { printf '[OK]     %s\n' "$*"; }
info() { printf '[INFO]   %s\n' "$*"; }
err()  { printf '[FEHLER] %s\n' "$*" >&2; }

reachable() { "$PY" - "$1" <<'EOF' >/dev/null 2>&1
import sys, urllib.request
urllib.request.urlopen(sys.argv[1], timeout=2)
EOF
}

command -v "$PY" >/dev/null 2>&1 || { err "Python 3 nicht gefunden (PYTHON=$PY)."; exit 1; }
ok "Python $("$PY" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"

if ! command -v ollama >/dev/null 2>&1; then
  err "Ollama nicht gefunden – Installation: https://ollama.com/download"
  exit 1
fi
if ! reachable "$OLLAMA_URL/api/tags"; then
  info "Ollama läuft noch nicht – wird gestartet …"
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  for _ in $(seq 1 20); do reachable "$OLLAMA_URL/api/tags" && break; sleep 1; done
  reachable "$OLLAMA_URL/api/tags" || { err "Ollama antwortet nicht (siehe /tmp/ollama.log)."; exit 1; }
fi
ok "Ollama erreichbar"

if ! "$PY" -m obito doctor; then
  echo
  echo "[HINWEIS] Der Systemcheck meldet Probleme. Modelle laden: ollama pull qwen2.5:7b && ollama pull nomic-embed-text"
  read -r -p "Trotzdem starten? [j/N] " antwort
  [[ "${antwort,,}" == j* ]] || exit 1
fi

if ! reachable "$OBITO_URL/api/status"; then
  info "Starte OBITO-Server unter $OBITO_URL …"
  nohup "$PY" -m obito serve >/tmp/obito-server.log 2>&1 &
  for _ in $(seq 1 30); do reachable "$OBITO_URL/api/status" && break; sleep 1; done
  reachable "$OBITO_URL/api/status" || { err "Server antwortet nicht (siehe /tmp/obito-server.log)."; exit 1; }
else
  info "OBITO-Server läuft bereits."
fi
ok "OBITO bereit: $OBITO_URL"

if command -v xdg-open >/dev/null 2>&1; then xdg-open "$OBITO_URL" >/dev/null 2>&1 &
elif command -v open >/dev/null 2>&1; then open "$OBITO_URL"
else info "Öffne im Browser: $OBITO_URL"; fi
echo "OBITO läuft. Server-Log: /tmp/obito-server.log"
