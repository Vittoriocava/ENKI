import argparse
import base64
import io
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg') # Prevents trying to open a display window
import matplotlib.pyplot as plt
import numpy as np
import uvicorn
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Aggiungiamo i percorsi del progetto per poter importare i moduli interni
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GRID_SIZE
from processing.tensor_builder import load_static_channels
from download.precipitation import create_precipitation_channels
from generate_dataset import generate_static_data

app = FastAPI(
    title="Live Channel Data Retriever",
    description="API for live fetching and returning requested channels as Base64 images for website visualization"
)

# Aggiunto CORS per consentire l'accesso da un frontend/sito web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mappatura della richiesta utente ai nomi dei canali interni
canali_info = [
    ('Altitude', 'terrain', 'altitude'),
    ('Slope', 'magma', 'slope'),
    ('Impermeability', 'gray', 'permeability'),
    ('Vegetation (NDVI)', 'Greens', 'vegetation'),
    ('Water Sources', 'cool', 'water_distance'),
    ('Rain 2 Days Ago', 'Blues', 'precip_day_before'),
    ('Rain Yesterday', 'Blues', 'precip_yesterday'),
    ('Rain Today', 'Blues', 'precip_today')
]

def load_or_generate_static():
    """Carica i canali statici, se non esistono li scarica"""
    static_channels = load_static_channels()
    # Verifica che la matrice non sia composta solo da zeri ignorando i runtime warnings
    is_empty = True
    if "altitude" in static_channels:
        if isinstance(static_channels["altitude"], np.ndarray):
            is_empty = np.all(static_channels["altitude"] == 0)
            
    if is_empty:
        print("[API] Dati statici mancanti. Avvio download (potrebbe richedere tempo)...")
        static_channels = generate_static_data()
        
    return static_channels


@app.get("/api/live")
def get_live_data(date: str = Query(..., description="Date in YYYY-MM-DD format")):
    """
    Endpoint per recuperare i canali specificati data una certa data.
    Restituisce un array di immagini PNG codificate in Base64 (data URI).
    """
    try:
        # Parsiamo la data. Impostiamo ore 12:00 di default come fa il generate_dataset.py
        target_date = datetime.strptime(date, "%Y-%m-%d") + timedelta(hours=12)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # 1. Caricamento dati statici (se già presenti non riscarica)
    static_channels = load_or_generate_static()

    # 2. Estrazione dati evento live per la data richiesta
    try: # scarica e/o interpola array dai netcfd originali per i valori meteo
        precip_channels = create_precipitation_channels(target_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving precipitation: {e}")

    # Unisci tutti i canali
    all_data = {**static_channels, **precip_channels}
    
    response_data = []

    # 3. Processa ciascun canale e generalo come immagine Base64
    for title, cmap, key in canali_info:
        data_matrix = all_data.get(key)
        
        if data_matrix is None:
            continue
            
        # Pulisce i dati da NaN che matplotlib potrebbe non renderizzare correttamente
        data_matrix = np.nan_to_num(data_matrix, nan=0.0)

        # Crea un'immagine utilizzando matplotlib per generare le mappe di colore corrette
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(data_matrix, cmap=cmap)
        ax.axis('off')
        
        # Salva direttamente l'output matplotlib in formato testo (Base64) bufferizzato nel server
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        response_data.append({
            "title": title,
            "key": key,
            "colormap": cmap,
            "image": f"data:image/png;base64,{img_base64}"
        })

    return {
        "status": "success",
        "date": date,
        "channels": response_data
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Data API server")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()
    
    print(f"API Live Server is running at http://{args.host}:{args.port}")
    print(f"To query: http://{args.host}:{args.port}/api/live?date=2023-11-05")
    
    # Lancia esplicitamente l'app attraverso uvicorn
    uvicorn.run("live_retriever:app", host=args.host, port=args.port, reload=True)
