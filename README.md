# ENKI

ENKI Dashboard — a Django-based flood risk dashboard with ResUNet flood prediction via Keras/Torch and A* routing on an OSMnx/NetworkX road graph.

## Requirements

- Python >= 3.14

## Setup

### With uv

```bash
git clone git@github.com:Vittoriocava/ENKI.git
cd ENKI
uv sync
uv run uvicorn enki_project.asgi:application --host 0.0.0.0 --port 8000
```

### With pip

```bash
git clone git@github.com:Vittoriocava/ENKI.git
cd ENKI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn enki_project.asgi:application --host 0.0.0.0 --port 8000
```

## Start the app

```bash
uv run uvicorn enki_project.asgi:application --host 0.0.0.0 --port 8000
```
