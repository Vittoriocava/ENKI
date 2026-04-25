import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import sys

# Aggiungiamo il path di config
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from config import CHANNEL_NAMES, TENSOR_DIR
except ImportError:
    print("Errore: impossibile importare config.py")
    sys.exit(1)

def visualize_sample(sample_id):
    index_path = TENSOR_DIR / "dataset_index.json"
    if not index_path.exists():
        print(f"Errore: {index_path} non trovato.")
        return
        
    with open(index_path, 'r') as f:
        index = json.load(f)
        
    # Trova il sample corrispondente
    sample_info = None
    if isinstance(sample_id, int):
        sample_id_str = f"{sample_id:06d}"
        for item in index:
            if item["sample_id"] == sample_id_str:
                sample_info = item
                break
    else:
        # Se è una stringa cerchiamo il match esatto
        for item in index:
            if item["sample_id"] == sample_id:
                sample_info = item
                break
                
    if not sample_info:
        print(f"Sample {sample_id} non trovato nell'indice.")
        return
        
    input_path = Path(sample_info["input_path"])
    target_path = Path(sample_info["target_path"])
    
    if not input_path.exists() or not target_path.exists():
        print(f"I file per il sample {sample_id} non sono presenti sul disco.")
        return
        
    print(f"Caricamento sample {sample_id}...")
    x = np.load(input_path)  
    y = np.load(target_path) 
    
    # Gestisco il caso in cui la forma sia (12, 512, 512) invece di (512, 512, 12)
    if x.shape[0] == len(CHANNEL_NAMES):
        x = np.transpose(x, (1, 2, 0))
    if y.ndim == 3 and y.shape[0] == 1:
        y = y[0]
        
    print(f"Shape input: {x.shape}")
    print(f"Shape target: {y.shape}")
    
    num_channels = len(CHANNEL_NAMES)
    
    # Crea una figure per plot dinamici: 12 input + 1 target = 13 subplot (griglia 4x4)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(f"Visualizzazione Sample ID: {sample_info['sample_id']}", fontsize=16)
    
    axes = axes.flatten()
    
    # Plot dei canali input
    for i in range(num_channels):
        ax = axes[i]
        channel_data = x[:, :, i]
        
        # Scegliamo colormap in base al tipo di dato (per bellezza)
        cmap = 'viridis'
        if 'precip' in CHANNEL_NAMES[i] or 'water' in CHANNEL_NAMES[i]:
            cmap = 'Blues'
        elif 'vegetation' in CHANNEL_NAMES[i]:
            cmap = 'Greens'
        elif 'soil' in CHANNEL_NAMES[i]:
            cmap = 'Oranges'
        elif 'permeability' in CHANNEL_NAMES[i]:
            cmap = 'Greys'
        elif 'altitude' in CHANNEL_NAMES[i] or 'slope' in CHANNEL_NAMES[i]:
            cmap = 'terrain'
            
        im = ax.imshow(channel_data, cmap=cmap)
        ax.set_title(f"Ch {i}: {CHANNEL_NAMES[i]}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    # Plot target (Flood mask)
    ax_target = axes[num_channels]
    im_target = ax_target.imshow(y, cmap='Reds', interpolation='nearest')
    ax_target.set_title("TARGET: Flood Mask")
    ax_target.axis('off')
    fig.colorbar(im_target, ax=ax_target, fraction=0.046, pad=0.04)
    
    # Rimuovi subplot in eccesso (ne usiamo 13, la griglia è 16)
    for i in range(num_channels + 1, 16):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizzatore dei tensor dataset generati")
    parser.add_argument("--sample", "-s", type=int, default=0, help="ID del sample da visualizzare (default: 0)")
    parser.add_argument("--random", "-r", action="store_true", help="Seleziona un sample casuale")
    args = parser.parse_args()
    
    if args.random:
        index_path = TENSOR_DIR / "dataset_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
            if index:
                import random
                sample = random.choice(index)
                visualize_sample(sample['sample_id'])
            else:
                print("L'indice del dataset è vuoto.")
        else:
            print("dataset_index.json non trovato.")
    else:
        visualize_sample(args.sample)
