"""
Utility di visualizzazione per tensori, canali e predizioni.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHANNEL_NAMES, GRID_SIZE


def plot_single_channel(
    data: np.ndarray,
    title: str = "Channel",
    cmap: str = "viridis",
    save_path: Optional[str] = None,
    vmin: float = None,
    vmax: float = None,
    figsize: tuple = (8, 8)
):
    """Visualizza un singolo canale come immagine."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Pixel X (40m)")
    ax.set_ylabel("Pixel Y (40m)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_all_channels(
    tensor: np.ndarray,
    channel_names: list = None,
    save_path: Optional[str] = None,
    figsize: tuple = (24, 20)
):
    """
    Visualizza tutti i canali di un tensore (C, H, W).

    Args:
        tensor: shape (C, H, W) o (H, W) per singolo canale
        channel_names: lista nomi canali
        save_path: percorso per salvare l'immagine
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES

    if tensor.ndim == 2:
        tensor = tensor[np.newaxis, ...]

    n_channels = tensor.shape[0]
    cols = 4
    rows = (n_channels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    cmaps = {
        "altitude": "terrain",
        "slope": "YlOrRd",
        "permeability": "RdYlGn",
        "vegetation": "Greens",
        "water_distance": "Blues_r",
        "soil_state": "gray",
        "soil_moisture": "YlGnBu",
    }

    for i in range(n_channels):
        name = channel_names[i] if i < len(channel_names) else f"Channel {i}"
        cmap = cmaps.get(name, "viridis")
        ax = axes[i]
        im = ax.imshow(tensor[i], cmap=cmap, origin="upper")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Nascondi assi vuoti
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Input Tensor Channels (512×512 @ 40m)", fontsize=16, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_prediction_overlay(
    prediction: np.ndarray,
    target: np.ndarray = None,
    dem: np.ndarray = None,
    save_path: Optional[str] = None,
    threshold: float = 0.5,
    figsize: tuple = (16, 6)
):
    """
    Overlay della predizione sulla mappa.

    Args:
        prediction: heatmap (H, W) con valori [0, 1]
        target: maschera target binaria (H, W)
        dem: DEM per sfondo
        save_path: percorso per salvare
        threshold: soglia per binarizzare la predizione
    """
    n_plots = 2 + (1 if target is not None else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    # Heatmap predizione
    ax = axes[0]
    if dem is not None:
        ax.imshow(dem, cmap="terrain", alpha=0.5, origin="upper")
    im = ax.imshow(prediction, cmap="YlOrRd", alpha=0.7, vmin=0, vmax=1, origin="upper")
    ax.set_title("Predizione Heatmap", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Predizione binarizzata
    ax = axes[1]
    binary_pred = (prediction > threshold).astype(np.float32)
    ax.imshow(binary_pred, cmap="Reds", origin="upper")
    ax.set_title(f"Predizione (soglia={threshold})", fontsize=12, fontweight="bold")

    # Target (se disponibile)
    if target is not None:
        ax = axes[2]
        ax.imshow(target, cmap="Reds", origin="upper")
        ax.set_title("Target (Ground Truth)", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_training_metrics(
    train_losses: list,
    val_losses: list,
    train_ious: list = None,
    val_ious: list = None,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5)
):
    """
    Plot delle curve di training.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss
    ax = axes[0]
    ax.plot(train_losses, label="Train Loss", color="#e74c3c", linewidth=2)
    ax.plot(val_losses, label="Val Loss", color="#3498db", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE + Dice)")
    ax.set_title("Loss Curve", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # IoU
    ax = axes[1]
    if train_ious and val_ious:
        ax.plot(train_ious, label="Train IoU", color="#e74c3c", linewidth=2)
        ax.plot(val_ious, label="Val IoU", color="#3498db", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("IoU Curve", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
