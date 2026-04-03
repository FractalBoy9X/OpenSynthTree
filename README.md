# OpenSynthTree — Docker Open-Source

A containerized toolkit for chemical molecule analysis.

Combines:
- **MolT5** — SMILES-to-text and text-to-SMILES translation
- **AiZynthFinder** — retrosynthesis using USPTO models
- **rxn-insight** — reaction classification, functional groups, by-products, conditions
- **PubChem API** — molecule lookup by name
- **RDKit** — SMILES validation and 2D structure visualization

## Requirements

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- **~16 GB of free disk space** (models + data):
  - ~790 MB — AiZynthFinder models (USPTO ONNX)
  - ~1.5 GB — rxn-insight database (Zenodo)
  - ~6.3 GB — MolT5 HuggingFace models (cache)
  - ~4 GB — Docker image (Python + dependencies)
- **Internet connection** (model downloads, PubChem API)

## Quick start

```bash
# 1. Build the Docker image
cd OpenSynthTree
docker compose build

# 2. Download data (AiZynthFinder models + rxn-insight database)
#    Data is saved to ./data/ (persistent, only needed once)
docker compose run --rm app download

# 3. Launch the interactive console
docker compose run --rm app

# 4. Or run the batch script (predefined molecules)
docker compose run --rm app batch
```

## Run modes

| Command | Description |
|---|---|
| `docker compose run --rm app` | Interactive console (REPL) |
| `docker compose run --rm app console` | Same as above (explicit mode) |
| `docker compose run --rm app batch` | Batch script |
| `docker compose run --rm app download` | Download data |
| `docker compose run --rm app shell` | Bash shell (debug) |

## Console commands

| Command | Description |
|---|---|
| `<name>` | Molecule name (e.g. `aspirin`) — PubChem SMILES lookup — MolT5 caption + PNG |
| `smiles <SMILES>` | Direct SMILES — MolT5 caption + visualization |
| `caption <description>` | Text description — generate SMILES (MolT5) |
| `retro <name/SMILES> [N]` | AiZynthFinder retrosynthesis — top N routes + rxn-insight analysis |
| `help` | List commands |
| `quit` / `exit` / `q` | Exit |

## Data structure (volumes)

Data is mounted from the host into the container as volumes — persistent across runs:

```
OpenSynthTree/
├── data/                           # created automatically
│   ├── aizynth_data/               # USPTO models (~790 MB)
│   │   ├── config.yml
│   │   ├── uspto_model.onnx
│   │   ├── uspto_templates.csv.gz
│   │   ├── uspto_ringbreaker_model.onnx
│   │   ├── uspto_ringbreaker_templates.csv.gz
│   │   ├── uspto_filter_model.onnx
│   │   └── zinc_stock.hdf5
│   ├── rxn_insight_data/           # reaction database (~1.5 GB)
│   │   └── uspto_rxn_insight.gzip
│   └── huggingface_cache/          # MolT5 models (~6.3 GB)
└── output/                         # PNG results
```

## Project structure

```
OpenSynthTree/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── entrypoint.sh               # container entrypoint
├── download_data.sh            # data download script
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT
├── THIRD-PARTY-NOTICES.md      # third-party licenses
├── README.md                   # this file
└── src/
    ├── console.py              # interactive console (REPL)
    ├── main.py                 # batch script
    └── aizynth_data/
        └── config.yml          # container path configuration
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `RETROCHEM_DEVICE` | `cpu` | Compute device (`cpu`, `cuda`) |
| `RETROCHEM_DATA_DIR` | `/app/data` | Data path inside the container |
| `RETROCHEM_OUTPUT_DIR` | `/app/output` | PNG output path |

## Using existing data

If you already have downloaded data (e.g. from a previous installation), copy it to the appropriate folders:

```bash
# AiZynthFinder models
cp -r /path/to/aizynth_data/* ./data/aizynth_data/

# rxn-insight database
cp /path/to/uspto_rxn_insight.gzip ./data/rxn_insight_data/
```

## GPU (CUDA)

To use an NVIDIA GPU, modify `docker-compose.yml`:

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

And build the image with a CUDA base image — change the first line of the `Dockerfile` to e.g.:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS base
```

## License

MIT License — see [LICENSE](LICENSE).

Third-party licenses — see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
