#!/bin/bash
set -e

# ─── Skrypt pobierania danych dla retro_chem ─────────────────────────────────
# Pobiera:
#   1. Modele AiZynthFinder (USPTO ONNX + szablony + ZINC stock)
#   2. Bazę rxn-insight (USPTO reakcje z Zenodo)
#
# Dane zapisywane do /app/data/ (mapowane jako volume na hosta)

CYAN='\033[96m'
GREEN='\033[92m'
YELLOW='\033[93m'
RED='\033[91m'
RESET='\033[0m'
BOLD='\033[1m'

DATA_DIR="${RETROCHEM_DATA_DIR:-/app/data}"
AIZYNTH_DIR="$DATA_DIR/aizynth_data"
RXN_DIR="$DATA_DIR/rxn_insight_data"

echo -e "${BOLD}${CYAN}══════════════════════════════════════════${RESET}"
echo -e "${BOLD}${CYAN}  retro_chem — Pobieranie danych${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════${RESET}"
echo ""

# ─── 1. AiZynthFinder ────────────────────────────────────────────────────────
echo -e "${BOLD}[1/2] Modele AiZynthFinder (USPTO)${RESET}"

if [ -f "$AIZYNTH_DIR/uspto_model.onnx" ] && \
   [ -f "$AIZYNTH_DIR/uspto_templates.csv.gz" ] && \
   [ -f "$AIZYNTH_DIR/zinc_stock.hdf5" ]; then
    echo -e "${GREEN}✓ Dane AiZynthFinder już istnieją — pomijam.${RESET}"
else
    echo -e "${CYAN}  Pobieranie przez aizynthfinder.tools.download_public_data...${RESET}"
    echo -e "${YELLOW}  (To może potrwać kilka minut — ~790 MB)${RESET}"
    mkdir -p "$AIZYNTH_DIR"
    python -m aizynthfinder.tools.download_public_data "$AIZYNTH_DIR"

    # Skopiuj config.yml ze ścieżkami kontenerowymi
    cp /app/src/aizynth_data/config.yml "$AIZYNTH_DIR/config.yml"
    echo -e "${GREEN}✓ Dane AiZynthFinder pobrane.${RESET}"
fi

echo ""

# ─── 2. rxn-insight (Zenodo) ─────────────────────────────────────────────────
echo -e "${BOLD}[2/2] Baza rxn-insight (USPTO z Zenodo)${RESET}"

RXN_FILE="$RXN_DIR/uspto_rxn_insight.gzip"
if [ -f "$RXN_FILE" ]; then
    echo -e "${GREEN}✓ Baza rxn-insight już istnieje — pomijam.${RESET}"
else
    echo -e "${CYAN}  Pobieranie z Zenodo...${RESET}"
    echo -e "${YELLOW}  (To może potrwać kilka-kilkanaście minut — ~1.5 GB)${RESET}"
    mkdir -p "$RXN_DIR"
    curl -L --progress-bar -o "$RXN_FILE" \
        "https://zenodo.org/api/records/10171745/files/uspto_rxn_insight.gzip/content"
    echo -e "${GREEN}✓ Baza rxn-insight pobrana.${RESET}"
fi

echo ""

# ─── Podsumowanie ─────────────────────────────────────────────────────────────
echo -e "${BOLD}${GREEN}══════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  Dane gotowe!${RESET}"
echo -e "${GREEN}  AiZynthFinder: $AIZYNTH_DIR/${RESET}"
echo -e "${GREEN}  rxn-insight:   $RXN_DIR/${RESET}"
echo -e "${BOLD}${GREEN}══════════════════════════════════════════${RESET}"
echo ""
echo -e "${CYAN}Modele MolT5 HuggingFace (~6.3 GB) zostaną pobrane${RESET}"
echo -e "${CYAN}automatycznie przy pierwszym uruchomieniu konsoli.${RESET}"
echo -e "${CYAN}Cache: /root/.cache/huggingface/ (mapowany jako volume)${RESET}"
