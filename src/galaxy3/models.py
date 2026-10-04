"""GalaxyCNN: CNN propia, entrenada desde cero, con BatchNorm y Dropout configurables.

Este módulo no conoce optimizer, loaders, MLflow ni training loops.

Los nombres de los submódulos son estables entre las configuraciones A, B y C
(``features.block1.conv``, ``features.block1.norm``, ..., ``classifier.dropout``):
``norm`` es ``BatchNorm2d`` o ``Identity`` y ``dropout`` es ``Dropout`` o ``Identity``.
El engine no depende de estos nombres; solo los usa la inspección pedagógica.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

FEATURE_CHANNELS: tuple[int, ...] = (32, 64, 128, 256)
POOLED_SIZE: tuple[int, int] = (2, 2)
HIDDEN_UNITS: int = 128


def _conv_block(in_channels: int, out_channels: int, use_batchnorm: bool) -> nn.Sequential:
    """Bloque Conv(3x3, padding=1) -> [BN | Identity] -> ReLU -> MaxPool(2, 2).

    Conserva la resolución en la convolución y la reduce a la mitad en el pooling:
    ``(B, in_channels, H, W) -> (B, out_channels, H/2, W/2)``.

    Con BN la convolución no lleva bias: el ``beta`` de BN cumple ese papel y el bias
    se cancelaría al restar la media del batch.
    """
    return nn.Sequential(
        OrderedDict(
            conv=nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm
            ),
            norm=nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            act=nn.ReLU(),
            pool=nn.MaxPool2d(kernel_size=2, stride=2),
        )
    )


class GalaxyCNN(nn.Module):
    """CNN de 4 bloques convolucionales para clasificar galaxias.

    Arquitectura (con entrada 128x128)::

        (B, 3, 128, 128)
        features.block1 -> (B,  32, 64, 64)
        features.block2 -> (B,  64, 32, 32)
        features.block3 -> (B, 128, 16, 16)
        features.block4 -> (B, 256,  8,  8)
        pool (AdaptiveAvgPool2d(2, 2)) -> (B, 256, 2, 2)
        classifier: Flatten -> Linear(1024, 128) -> ReLU -> [Dropout | Identity]
                    -> Linear(128, num_classes) -> (B, num_classes)

    Args:
        num_classes: Número de clases de salida (>= 2), igual que ``ExperimentConfig``.
        use_batchnorm: Si es ``True``, cada bloque usa ``BatchNorm2d`` y convoluciones
            sin bias; si es ``False``, ``Identity`` y convoluciones con bias.
        dropout_p: Probabilidad de Dropout en el clasificador, en ``[0, 1)``. Con
            ``0.0`` se usa ``Identity``.

    Raises:
        ValueError: Si ``num_classes < 2`` o ``dropout_p`` está fuera de ``[0, 1)``.
    """

    def __init__(
        self,
        num_classes: int = 3,
        use_batchnorm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes debe ser >= 2; se recibió {num_classes}")
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p debe estar en [0, 1); se recibió {dropout_p}")

        in_channels = (3, *FEATURE_CHANNELS[:-1])
        self.features = nn.Sequential(
            OrderedDict(
                (f"block{i}", _conv_block(c_in, c_out, use_batchnorm))
                for i, (c_in, c_out) in enumerate(
                    zip(in_channels, FEATURE_CHANNELS, strict=True), start=1
                )
            )
        )
        self.pool = nn.AdaptiveAvgPool2d(POOLED_SIZE)
        flat_features = FEATURE_CHANNELS[-1] * POOLED_SIZE[0] * POOLED_SIZE[1]
        self.classifier = nn.Sequential(
            OrderedDict(
                flatten=nn.Flatten(),
                fc1=nn.Linear(flat_features, HIDDEN_UNITS),
                act=nn.ReLU(),
                dropout=nn.Dropout(dropout_p) if dropout_p > 0.0 else nn.Identity(),
                fc2=nn.Linear(HIDDEN_UNITS, num_classes),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula los logits.

        Args:
            x: Tensor ``(B, 3, H, W)``; la arquitectura está pensada para ``H = W = 128``.

        Returns:
            Logits ``(B, num_classes)``, sin softmax (``CrossEntropyLoss`` lo aplica).
        """
        return self.classifier(self.pool(self.features(x)))


def count_parameters(model: nn.Module) -> int:
    """Número total de parámetros (los buffers, como las running stats de BN, no cuentan)."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Número de parámetros con ``requires_grad=True``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
