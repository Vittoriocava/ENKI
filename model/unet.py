"""
UNet Architecture per segmentazione semantica di alluvioni.

Architettura standard encoder-decoder con skip connections:
- Encoder: 4 blocchi di downsampling
- Bottleneck: blocco centrale
- Decoder: 4 blocchi di upsampling con concatenazione skip
- Output: 1 canale con attivazione Sigmoid

Input:  (B, 12, 512, 512)
Output: (B, 1, 512, 512) — probabilità alluvione [0, 1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NUM_INPUT_CHANNELS, UNET_BASE_FILTERS, UNET_DEPTH


class ConvBlock(nn.Module):
    """
    Blocco convoluzionale doppio: (Conv3x3 → BN → ReLU) × 2
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    Blocco encoder: ConvBlock → MaxPool2x2
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (pooled, skip) — pooled per il livello successivo, skip per il decoder
        """
        skip = self.conv(x)
        pooled = self.pool(skip)
        return pooled, skip


class DecoderBlock(nn.Module):
    """
    Blocco decoder: Upsample → Concat skip → ConvBlock
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: feature map dal livello inferiore
            skip: skip connection dal corrispondente encoder

        Returns:
            feature map decodificata
        """
        x = self.up(x)

        # Gestisci dimensioni diverse (padding/cropping)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [
            diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2
        ])

        # Concatena skip connection
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet per predizione alluvioni.

    Architettura:
        Encoder: 12 → 64 → 128 → 256 → 512
        Bottleneck: 512 → 1024
        Decoder: 1024 → 512 → 256 → 128 → 64
        Output: 64 → 1 (Sigmoid)

    Args:
        in_channels: numero canali input (default: 12)
        out_channels: numero canali output (default: 1)
        base_filters: filtri del primo livello (default: 64)
        depth: profondità encoder/decoder (default: 4)
    """

    def __init__(
        self,
        in_channels: int = NUM_INPUT_CHANNELS,
        out_channels: int = 1,
        base_filters: int = UNET_BASE_FILTERS,
        depth: int = UNET_DEPTH,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        # Calcola il numero di filtri per ogni livello
        filters = [base_filters * (2 ** i) for i in range(depth + 1)]
        # [64, 128, 256, 512, 1024]

        # ── Encoder ──
        self.encoders = nn.ModuleList()
        enc_in = in_channels
        for i in range(depth):
            self.encoders.append(EncoderBlock(enc_in, filters[i]))
            enc_in = filters[i]

        # ── Bottleneck ──
        self.bottleneck = ConvBlock(filters[depth - 1], filters[depth])

        # ── Decoder ──
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.decoders.append(DecoderBlock(filters[i + 1], filters[i]))

        # ── Output ──
        self.output_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: input tensor (B, 12, 512, 512)

        Returns:
            probabilità alluvione (B, 1, 512, 512) in [0, 1]
        """
        # Encoder
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (skip connections in ordine inverso)
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = decoder(x, skip)

        # Output con Sigmoid
        x = self.output_conv(x)
        x = torch.sigmoid(x)

        return x

    def count_parameters(self) -> int:
        """Conta i parametri totali del modello."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    in_channels: int = NUM_INPUT_CHANNELS,
    device: str = "auto"
) -> tuple[UNet, torch.device]:
    """
    Factory function per creare il modello UNet.

    Args:
        in_channels: canali di input
        device: "auto", "cuda", "cpu"

    Returns:
        (model, device)
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = UNet(in_channels=in_channels)
    model = model.to(device)

    n_params = model.count_parameters()
    print(f"[UNET] Modello creato su {device}")
    print(f"[UNET] Parametri: {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"[UNET] Input:  ({in_channels}, 512, 512)")
    print(f"[UNET] Output: (1, 512, 512)")

    return model, device


if __name__ == "__main__":
    model, device = create_model()

    # Test forward pass
    dummy_input = torch.randn(1, NUM_INPUT_CHANNELS, 512, 512).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n[TEST] Forward pass:")
    print(f"  Input:  {dummy_input.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Range:  [{output.min():.4f}, {output.max():.4f}]")
    assert output.shape == (1, 1, 512, 512), f"Shape errata: {output.shape}"
    assert output.min() >= 0 and output.max() <= 1, "Output fuori range [0, 1]"
    print("  ✓ Test superato!")
