"""
Loss functions per il training della UNet.

Combinazione di Binary Cross-Entropy (BCE) e Dice Loss:
- BCE: loss standard per classificazione binaria pixel-wise
- Dice Loss: penalizza i falsi negativi, ideale per dati sbilanciati
  (pochi pixel alluvione vs molti pixel asciutti)

Loss totale = α * BCE + β * (1 - Dice)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BCE_WEIGHT, DICE_WEIGHT, DICE_SMOOTH


class DiceLoss(nn.Module):
    """
    Dice Loss per segmentazione binaria.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    DiceLoss = 1 - Dice

    Il smooth factor previene divisione per zero e stabilizza il training.

    Args:
        smooth: fattore di smoothing (default: 1.0)
    """

    def __init__(self, smooth: float = DICE_SMOOTH):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: predizioni (B, 1, H, W) in [0, 1]
            target: target (B, 1, H, W) binario {0, 1}

        Returns:
            Dice Loss scalare
        """
        # Flatten
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        # Intersezione
        intersection = (pred_flat * target_flat).sum()

        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combinazione pesata di BCE Loss e Dice Loss.

    Loss = α * BCE(pred, target) + β * DiceLoss(pred, target)

    Args:
        bce_weight: peso α per la componente BCE
        dice_weight: peso β per la componente Dice
        smooth: smooth factor per Dice
    """

    def __init__(
        self,
        bce_weight: float = BCE_WEIGHT,
        dice_weight: float = DICE_WEIGHT,
        smooth: float = DICE_SMOOTH
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: predizioni (B, 1, H, W) in [0, 1] (dopo Sigmoid)
            target: target (B, 1, H, W) binario {0, 1}

        Returns:
            Loss combinata scalare
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss


class FocalDiceLoss(nn.Module):
    """
    Variante con Focal Loss al posto di BCE per gestire
    lo sbilanciamento estremo (pochi pixel alluvione).

    FocalLoss = -α * (1 - p)^γ * log(p)

    Args:
        alpha: peso per la classe positiva
        gamma: esponente di focusing
        dice_weight: peso della componente Dice
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        dice_weight: float = DICE_WEIGHT,
        smooth: float = DICE_SMOOTH
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Focal Loss
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = (focal_weight * bce).mean()

        # Dice
        dice_loss = self.dice(pred, target)

        return focal_loss + self.dice_weight * dice_loss


# ─────────────────────────────────────────────
# Metriche di valutazione
# ─────────────────────────────────────────────
def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calcola Intersection over Union (IoU / Jaccard Index).

    Args:
        pred: predizioni (B, 1, H, W) in [0, 1]
        target: target (B, 1, H, W) binario
        threshold: soglia per binarizzare pred

    Returns:
        IoU scalare
    """
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection

    if union == 0:
        return 1.0  # Entrambi vuoti → perfetto

    return float(intersection / union)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calcola Dice coefficient."""
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    total = pred_bin.sum() + target.sum()

    if total == 0:
        return 1.0

    return float(2 * intersection / total)


def compute_precision_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> tuple[float, float]:
    """Calcola Precision e Recall."""
    pred_bin = (pred > threshold).float()

    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()

    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))

    return precision, recall


if __name__ == "__main__":
    # Test delle loss
    torch.manual_seed(42)

    pred = torch.sigmoid(torch.randn(2, 1, 512, 512))
    target = (torch.rand(2, 1, 512, 512) > 0.9).float()  # ~10% positivi

    # BCE + Dice
    bce_dice = BCEDiceLoss()
    loss = bce_dice(pred, target)
    print(f"[TEST] BCEDiceLoss: {loss.item():.4f}")

    # Focal + Dice
    focal_dice = FocalDiceLoss()
    loss_focal = focal_dice(pred, target)
    print(f"[TEST] FocalDiceLoss: {loss_focal.item():.4f}")

    # Metriche
    iou = compute_iou(pred, target)
    dice = compute_dice(pred, target)
    prec, rec = compute_precision_recall(pred, target)
    print(f"[TEST] IoU={iou:.4f}, Dice={dice:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
    print("  ✓ Test superato!")
