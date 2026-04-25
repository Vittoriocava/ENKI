"""
Script di training per la UNet di predizione alluvioni.

Features:
- Training loop con BCE + Dice Loss
- Early stopping
- Checkpoint saving
- Logging metriche (loss, IoU, Dice, Precision, Recall)
- Supporto GPU/CPU
- TensorBoard logging opzionale

Uso: python train.py [--epochs N] [--batch-size B] [--lr LR]
"""
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    TENSOR_DIR, CHECKPOINTS_DIR, PROJECT_ROOT,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE, TRAIN_VAL_SPLIT,
    NUM_INPUT_CHANNELS, GRID_SIZE
)
from model.unet import UNet, create_model
from model.losses import (
    BCEDiceLoss, FocalDiceLoss,
    compute_iou, compute_dice, compute_precision_recall
)
from dataset.flood_dataset import FloodDataset, create_dataloaders
from utils.viz import plot_training_metrics, plot_prediction_overlay


class Trainer:
    """
    Trainer per la UNet di predizione alluvioni.
    """

    def __init__(
        self,
        model: UNet,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        checkpoint_dir: Path = CHECKPOINTS_DIR,
        patience: int = EARLY_STOPPING_PATIENCE,
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_ious = []
        self.val_ious = []
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.current_epoch = 0

    def train_epoch(self) -> dict:
        """Esegui un'epoca di training."""
        self.model.train()
        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_dice = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Train E{self.current_epoch}",
            leave=False
        )

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Metriche
            with torch.no_grad():
                iou = compute_iou(outputs, targets)
                dice = compute_dice(outputs, targets)

            epoch_loss += loss.item()
            epoch_iou += iou
            epoch_dice += dice
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "IoU": f"{iou:.4f}",
                "Dice": f"{dice:.4f}",
            })

        return {
            "loss": epoch_loss / max(n_batches, 1),
            "iou": epoch_iou / max(n_batches, 1),
            "dice": epoch_dice / max(n_batches, 1),
        }

    @torch.no_grad()
    def validate_epoch(self) -> dict:
        """Esegui un'epoca di validazione."""
        self.model.eval()
        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_dice = 0.0
        epoch_prec = 0.0
        epoch_rec = 0.0
        n_batches = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            iou = compute_iou(outputs, targets)
            dice = compute_dice(outputs, targets)
            prec, rec = compute_precision_recall(outputs, targets)

            epoch_loss += loss.item()
            epoch_iou += iou
            epoch_dice += dice
            epoch_prec += prec
            epoch_rec += rec
            n_batches += 1

        return {
            "loss": epoch_loss / max(n_batches, 1),
            "iou": epoch_iou / max(n_batches, 1),
            "dice": epoch_dice / max(n_batches, 1),
            "precision": epoch_prec / max(n_batches, 1),
            "recall": epoch_rec / max(n_batches, 1),
        }

    def save_checkpoint(self, filename: str = "best_model.pth"):
        """Salva checkpoint del modello."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename

        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_ious": self.train_ious,
            "val_ious": self.val_ious,
        }, path)

        print(f"  💾 Checkpoint salvato: {path}")

    def load_checkpoint(self, filename: str = "best_model.pth"):
        """Carica checkpoint."""
        path = self.checkpoint_dir / filename
        if not path.exists():
            print(f"  Nessun checkpoint trovato: {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.train_ious = checkpoint.get("train_ious", [])
        self.val_ious = checkpoint.get("val_ious", [])

        print(f"  ✓ Checkpoint caricato: epoch={self.current_epoch}, "
              f"best_loss={self.best_val_loss:.4f}")
        return True

    def train(self, num_epochs: int = NUM_EPOCHS):
        """
        Training loop completo.

        Args:
            num_epochs: numero massimo di epoche
        """
        print("╔══════════════════════════════════════════╗")
        print("║          TRAINING UNet ALLUVIONI         ║")
        print("╚══════════════════════════════════════════╝")
        print(f"  Device:     {self.device}")
        print(f"  Epochs:     {num_epochs}")
        print(f"  Batch size: {self.train_loader.batch_size}")
        print(f"  Train set:  {len(self.train_loader.dataset)} sample")
        print(f"  Val set:    {len(self.val_loader.dataset)} sample")
        print(f"  Patience:   {self.patience}")
        print()

        start_time = time.time()

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Training
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics["loss"])
            self.train_ious.append(train_metrics["iou"])

            # Validation
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics["loss"])
            self.val_ious.append(val_metrics["iou"])

            # Learning rate scheduling
            self.scheduler.step(val_metrics["loss"])

            epoch_time = time.time() - epoch_start

            # Logging
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1:3d}/{num_epochs} "
                f"│ Train Loss: {train_metrics['loss']:.4f} "
                f"│ Val Loss: {val_metrics['loss']:.4f} "
                f"│ Val IoU: {val_metrics['iou']:.4f} "
                f"│ Val Dice: {val_metrics['dice']:.4f} "
                f"│ P/R: {val_metrics['precision']:.2f}/{val_metrics['recall']:.2f} "
                f"│ LR: {current_lr:.2e} "
                f"│ {epoch_time:.1f}s"
            )

            # Early stopping check
            if val_metrics["loss"] < self.best_val_loss:
                improvement = self.best_val_loss - val_metrics["loss"]
                self.best_val_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
                self.save_checkpoint("best_model.pth")
                print(f"  ⬆️  Miglioramento: {improvement:.4f}")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"\n  ⏹️  Early stopping dopo {epoch + 1} epoche "
                          f"(nessun miglioramento per {self.patience} epoche)")
                    break

            # Checkpoint periodico
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1:03d}.pth")

        total_time = time.time() - start_time
        print(f"\n{'=' * 50}")
        print(f"Training completato in {total_time / 60:.1f} minuti")
        print(f"Miglior Val Loss: {self.best_val_loss:.4f}")

        # Salva metriche
        self._save_metrics()

        # Plot
        try:
            plot_training_metrics(
                self.train_losses, self.val_losses,
                self.train_ious, self.val_ious,
                save_path=str(self.checkpoint_dir / "training_curves.png")
            )
            print(f"  📊 Grafici salvati: {self.checkpoint_dir / 'training_curves.png'}")
        except Exception as e:
            print(f"  ⚠ Errore grafici: {e}")

    def _save_metrics(self):
        """Salva le metriche di training su file JSON."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "best_val_loss": self.best_val_loss,
            "total_epochs": len(self.train_losses),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_ious": self.train_ious,
            "val_ious": self.val_ious,
        }
        path = self.checkpoint_dir / "training_metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Training UNet alluvioni")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--resume", action="store_true", help="Riprendi da checkpoint")
    parser.add_argument("--loss", choices=["bce_dice", "focal_dice"],
                        default="bce_dice", help="Funzione di loss")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # ── Modello ──
    model, device = create_model(
        in_channels=NUM_INPUT_CHANNELS,
        device=args.device
    )

    # ── Loss ──
    if args.loss == "focal_dice":
        criterion = FocalDiceLoss()
        print("[LOSS] Usando FocalDiceLoss")
    else:
        criterion = BCEDiceLoss()
        print("[LOSS] Usando BCEDiceLoss")

    # ── Optimizer & Scheduler ──
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # ── Dataset ──
    try:
        train_loader, val_loader = create_dataloaders(
            tensor_dir=TENSOR_DIR,
            batch_size=args.batch_size,
            train_split=TRAIN_VAL_SPLIT,
            num_workers=args.num_workers
        )
    except ValueError as e:
        print(f"\n⚠ {e}")
        print("  Esegui prima: python generate_dataset.py")
        return

    # ── Trainer ──
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=CHECKPOINTS_DIR,
        patience=args.patience,
    )

    # Resume
    if args.resume:
        trainer.load_checkpoint()

    # ── Training ──
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
