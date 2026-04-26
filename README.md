# ENKI

ENKI Dashboard — a Django-based geospatial analysis tool built with OSMnx, Folium, and scikit-learn.

## Requirements

- Python >= 3.14

## Setup

### With uv

```bash
git clone git@github.com:Vittoriocava/ENKI.git
cd ENKI
uv sync
uv run manage.py runserver
```

### With pip

```bash
git clone git@github.com:Vittoriocava/ENKI.git
cd ENKI
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py runserver
```

> [!WARNING]
> This is a development environment and should only be used for demonstration purposes.
> Do not expose this service to the internet.
