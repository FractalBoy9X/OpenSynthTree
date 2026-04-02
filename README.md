# retro_chem — Docker Open-Source

Kontenerowa wersja narzędzia **retro_chem** do analizy cząsteczek chemicznych.

Łączy:
- **MolT5** — tłumaczenie SMILES na opisy tekstowe i odwrotnie
- **AiZynthFinder** — retrosynteza z użyciem modeli USPTO
- **rxn-insight** — klasyfikacja reakcji, grupy funkcyjne, produkty uboczne, warunki
- **PubChem API** — wyszukiwanie cząsteczek po nazwie
- **RDKit** — walidacja SMILES i wizualizacja struktur 2D

## Wymagania

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- **~16 GB wolnego miejsca** (modele + dane):
  - ~790 MB — modele AiZynthFinder (USPTO ONNX)
  - ~1.5 GB — baza rxn-insight (Zenodo)
  - ~6.3 GB — modele MolT5 HuggingFace (cache)
  - ~4 GB — obraz Docker (Python + zależności)
- **Połączenie z internetem** (pobieranie modeli, PubChem API)

## Szybki start

```bash
# 1. Zbuduj obraz Docker
cd contener_open_retrochem
docker compose build

# 2. Pobierz dane (modele AiZynthFinder + baza rxn-insight)
#    Dane zapisywane do ./data/ (persystentne, tylko za pierwszym razem)
docker compose run --rm app download

# 3. Uruchom konsolę interaktywną
docker compose run --rm app

# 4. Lub uruchom skrypt batchowy (predefiniowane cząsteczki)
docker compose run --rm app batch
```

## Tryby uruchamiania

| Komenda | Opis |
|---|---|
| `docker compose run --rm app` | Konsola interaktywna (REPL) |
| `docker compose run --rm app console` | j.w. (jawny tryb) |
| `docker compose run --rm app batch` | Skrypt batchowy |
| `docker compose run --rm app download` | Pobieranie danych |
| `docker compose run --rm app shell` | Powłoka bash (debug) |

## Komendy konsoli

| Polecenie | Opis |
|---|---|
| `<nazwa>` | Nazwa cząsteczki (np. `aspirin`) → PubChem SMILES → opis MolT5 + PNG |
| `smiles <SMILES>` | Bezpośredni SMILES → opis MolT5 + wizualizacja |
| `caption <opis>` | Opis tekstowy → wygenerowany SMILES (MolT5) |
| `retro <nazwa/SMILES> [N]` | Retrosynteza AiZynthFinder — top N tras + analiza rxn-insight |
| `help` | Lista poleceń |
| `quit` / `exit` / `q` | Wyjście |

## Struktura danych (volumes)

Dane są montowane z hosta do kontenera jako volumes — persystentne między uruchomieniami:

```
contener_open_retrochem/
├── data/                           # tworzony automatycznie
│   ├── aizynth_data/               # modele USPTO (~790 MB)
│   │   ├── config.yml
│   │   ├── uspto_model.onnx
│   │   ├── uspto_templates.csv.gz
│   │   ├── uspto_ringbreaker_model.onnx
│   │   ├── uspto_ringbreaker_templates.csv.gz
│   │   ├── uspto_filter_model.onnx
│   │   └── zinc_stock.hdf5
│   ├── rxn_insight_data/           # baza reakcji (~1.5 GB)
│   │   └── uspto_rxn_insight.gzip
│   └── huggingface_cache/          # modele MolT5 (~6.3 GB)
└── output/                         # wyniki PNG
```

## Struktura projektu

```
contener_open_retrochem/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── entrypoint.sh               # punkt wejścia kontenera
├── download_data.sh            # skrypt pobierania danych
├── requirements.txt            # zależności Python
├── LICENSE                     # MIT
├── THIRD-PARTY-NOTICES.md      # licencje zależności
├── README.md                   # ten plik
└── src/
    ├── console.py              # konsola interaktywna (REPL)
    ├── main.py                 # skrypt batchowy
    └── aizynth_data/
        └── config.yml          # konfiguracja ścieżek kontenerowych
```

## Zmienne środowiskowe

| Zmienna | Domyślna | Opis |
|---|---|---|
| `RETROCHEM_DEVICE` | `cpu` | Urządzenie obliczeniowe (`cpu`, `cuda`) |
| `RETROCHEM_DATA_DIR` | `/app/data` | Ścieżka do danych w kontenerze |
| `RETROCHEM_OUTPUT_DIR` | `/app/output` | Ścieżka do wyników PNG |

## Użycie z istniejącymi danymi

Jeśli masz już pobrane dane (np. z wcześniejszej instalacji), skopiuj je do odpowiednich folderów:

```bash
# Modele AiZynthFinder
cp -r /ścieżka/do/aizynth_data/* ./data/aizynth_data/

# Baza rxn-insight
cp /ścieżka/do/uspto_rxn_insight.gzip ./data/rxn_insight_data/
```

## GPU (CUDA)

Aby użyć GPU NVIDIA, zmień `docker-compose.yml`:

```yaml
services:
  app:
    # ...
    environment:
      - RETROCHEM_DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

I zbuduj obraz z bazą `pytorch/pytorch:*-cuda*-runtime`:
zmień pierwszą linię `Dockerfile` na np.:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS base
```

## Licencja

MIT License — patrz [LICENSE](LICENSE).

Licencje zależności — patrz [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
