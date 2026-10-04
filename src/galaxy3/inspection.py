"""Inspección pedagógica del modelo: captura de feature maps con forward hooks temporales.

No participa en el entrenamiento ni modifica el ``forward`` del modelo. Se basa en los
nombres estables de ``model.named_modules()`` (por ejemplo ``features.block1.act``), que
son iguales en las configuraciones A, B y C.
"""

from __future__ import annotations

import torch
from torch import nn


def capture_feature_maps(
    model: nn.Module,
    X_batch: torch.Tensor,
    layer_name: str,
    device: torch.device,
) -> torch.Tensor:
    """Devuelve la salida del submódulo ``layer_name`` al pasar ``X_batch`` por el modelo.

    Registra un forward hook temporal sobre ese submódulo, ejecuta un forward en modo
    eval bajo ``torch.inference_mode()`` y retira el hook en un bloque ``finally``, incluso
    si el forward falla. Al terminar, el modelo vuelve a su modo train/eval previo. No
    modifica pesos ni running statistics de BN.

    Capa recomendada: ``features.blockN.act`` (salida de la ReLU, antes del pooling).
    Con entrada 128x128: block1 -> ``(B, 32, 128, 128)``, block2 -> ``(B, 64, 64, 64)``,
    block3 -> ``(B, 128, 32, 32)``, block4 -> ``(B, 256, 16, 16)``.

    Args:
        model: Modelo ya movido a ``device``.
        X_batch: Batch normalizado ``(B, C, H, W)``; se transfiere a ``device``.
        layer_name: Nombre del submódulo según ``model.named_modules()``.
        device: Device del modelo.

    Returns:
        Activaciones ``(B, C', H', W')`` en CPU, desacopladas del grafo (no requieren
        gradiente), como tensor normal (no de inferencia).

    Raises:
        ValueError: Si ``X_batch`` no es 4D, si ``layer_name`` no existe en el modelo o si
            la salida de esa capa no es un tensor 4D.
        RuntimeError: Si la capa no se ejecutó durante el forward.
    """
    if X_batch.ndim != 4:
        raise ValueError(f"X_batch debe ser (B, C, H, W); tiene forma {tuple(X_batch.shape)}")
    modules = dict(model.named_modules())
    if not layer_name or layer_name not in modules:
        options = ", ".join(name for name in modules if name.endswith(".act"))
        raise ValueError(f"No existe la capa {layer_name!r}. Capas recomendadas: {options}")

    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured.append(output.detach())

    was_training = model.training
    handle = modules[layer_name].register_forward_hook(hook)
    try:
        model.eval()
        with torch.inference_mode():
            model(X_batch.to(device))
    finally:
        handle.remove()
        model.train(was_training)

    if not captured:
        raise RuntimeError(f"La capa {layer_name!r} no se ejecutó durante el forward")
    output = captured[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 4:
        raise ValueError(f"La salida de {layer_name!r} no es un tensor (B, C, H, W)")
    return output.cpu().clone()
