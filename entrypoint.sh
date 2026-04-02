#!/bin/bash
set -e

# ─── retro_chem Docker entrypoint ────────────────────────────────────────────
# Tryby:
#   console  — interaktywna konsola REPL (domyślny)
#   batch    — skrypt batchowy (predefiniowane cząsteczki)
#   download — pobieranie danych (modele AiZynthFinder + baza rxn-insight)
#   shell    — powłoka bash (debug)
#   *        — dowolna komenda

# Upewnij się, że config.yml jest na miejscu (jeśli volume jest pusty)
CONFIG_DST="/app/data/aizynth_data/config.yml"
CONFIG_SRC="/app/src/aizynth_data/config.yml"
if [ ! -f "$CONFIG_DST" ] && [ -f "$CONFIG_SRC" ]; then
    mkdir -p /app/data/aizynth_data
    cp "$CONFIG_SRC" "$CONFIG_DST"
    echo "Skopiowano config.yml do $CONFIG_DST"
fi

MODE="${1:-console}"

case "$MODE" in
    console)
        echo "Uruchamiam konsolę interaktywną..."
        exec python /app/src/console.py
        ;;
    batch)
        echo "Uruchamiam skrypt batchowy..."
        exec python /app/src/main.py
        ;;
    download)
        echo "Pobieranie danych..."
        exec /app/download_data.sh
        ;;
    shell)
        exec /bin/bash
        ;;
    *)
        exec "$@"
        ;;
esac
