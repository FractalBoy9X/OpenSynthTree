FROM python:3.11-slim AS base

# Metadane
LABEL maintainer="retro_chem contributors"
LABEL description="retro_chem — MolT5 + AiZynthFinder + rxn-insight (Docker)"
LABEL license="MIT"

# Zmienne środowiskowe
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    RETROCHEM_APP_DIR=/app \
    RETROCHEM_DATA_DIR=/app/data \
    RETROCHEM_OUTPUT_DIR=/app/output \
    RETROCHEM_DEVICE=cpu

# Zależności systemowe
#  - libxrender1, libxext6: wymagane przez RDKit do renderowania obrazów
#  - fonts-dejavu: fonty dla Pillow (generowanie PNG z tekstem)
#  - curl: pobieranie danych (download_data.sh)
#  - libhdf5-dev: obsługa plików HDF5 (zinc_stock.hdf5)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libsm6 \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    curl \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalacja zależności Python (osobna warstwa — cache)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install aizynthfinder>=4.0.0 --no-deps && \
    pip install -r requirements.txt

# Kopiowanie kodu źródłowego
COPY src/ /app/src/
COPY entrypoint.sh /app/entrypoint.sh
COPY download_data.sh /app/download_data.sh
RUN chmod +x /app/entrypoint.sh /app/download_data.sh

# Katalogi na dane i wyniki
RUN mkdir -p /app/data/aizynth_data \
             /app/data/rxn_insight_data \
             /app/output

# Kopiowanie config.yml do lokalizacji danych
# (zostanie nadpisany jeśli użytkownik zamontuje volume z danymi)
RUN cp /app/src/aizynth_data/config.yml /app/data/aizynth_data/config.yml

EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["console"]
